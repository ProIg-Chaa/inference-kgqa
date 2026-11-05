"""Train the pinned coetaur0/ESIM on MetaQA PHL and evaluate independent Dev.

Random trainable embeddings and small dimensions are explicit debug settings.
The upstream full sequential architecture is imported unchanged.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time

import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.esim_adapter import PHLDataset, collate_phl, create_model, model_inputs, qa_metrics


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def write_jsonl(path, rows):
    with path.open('w', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def original_question_ids(path, limit):
    selected = []
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            if limit is not None and len(selected) >= limit:
                break
            selected.append(json.loads(line)['question_id'])
    return set(selected)


def evaluate(model, loader, class_weights, threshold, question_file, question_ids):
    model.eval()
    predictions, correct, loss_sum, denominator = [], 0, 0.0, 0.0
    confusion = [[0, 0], [0, 0]]
    with torch.no_grad():
        for batch in loader:
            logits, probabilities = model(*model_inputs(batch, model.device))
            labels = batch['label'].to(model.device)
            if not torch.isfinite(logits).all():
                raise RuntimeError('Non-finite Dev logits; inspect current model/configuration.')
            loss_sum += nn.functional.cross_entropy(logits, labels, weight=class_weights, reduction='sum').item()
            denominator += class_weights[labels].sum().item()
            for i, score in enumerate(probabilities[:, 0].cpu().tolist()):
                gold = int(batch['label'][i])
                predicted = 0 if score >= threshold else 1
                correct += predicted == gold
                confusion[gold][predicted] += 1
                predictions.append(dict(sample_id=batch['sample_id'][i], question_id=batch['question_id'][i],
                                        candidate=batch['candidate'][i], entailment_probability=score,
                                        predicted_label=predicted, gold_label=gold))
    qa, details = qa_metrics(predictions, question_file, question_ids)
    tp, fp, fn = confusion[0][0], confusion[1][0], confusion[0][1]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    metrics = dict(loss=loss_sum / denominator, candidate_accuracy=correct / len(predictions),
                   candidate_samples=len(predictions), confusion_matrix_gold_rows=confusion,
                   entailment_precision=precision, entailment_recall=recall,
                   entailment_f1=2 * precision * recall / (precision + recall) if precision + recall else 0.0,
                   entailment_threshold=threshold, **qa)
    return metrics, predictions, details


def train_epoch(model, loader, optimizer, class_weights, max_norm):
    model.train()
    loss_sum, denominator, correct, samples = 0.0, 0.0, 0, 0
    for index, batch in enumerate(loader, 1):
        labels = batch['label'].to(model.device)
        optimizer.zero_grad(set_to_none=True)
        logits, probabilities = model(*model_inputs(batch, model.device))
        numerator = nn.functional.cross_entropy(logits, labels, weight=class_weights, reduction='sum')
        weights = class_weights[labels].sum()
        loss = numerator / weights
        if not torch.isfinite(loss):
            raise RuntimeError(f'Non-finite training loss at batch {index}')
        loss.backward()
        norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        if not torch.isfinite(norm):
            raise RuntimeError(f'Non-finite gradients at batch {index}')
        optimizer.step()
        loss_sum += numerator.item()
        denominator += weights.item()
        correct += (probabilities.argmax(-1) == labels).sum().item()
        samples += len(labels)
        if index % 100 == 0 or index == len(loader):
            print(f'  batch {index}/{len(loader)} loss={loss_sum / denominator:.6f}', flush=True)
    return dict(loss=loss_sum / denominator, candidate_accuracy=correct / samples, samples=samples, batches=len(loader))


def smoke(model, dataset, class_weights, config, output, source):
    batch = collate_phl([dataset[index] for index in range(min(8, len(dataset)))])
    before = model._classification[-1].weight.detach().clone()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    model.train()
    logits, probabilities = model(*model_inputs(batch, model.device))
    loss = nn.functional.cross_entropy(logits, batch['label'].to(model.device), weight=class_weights)
    loss.backward()
    gradient_groups = {}
    for name in ['_word_embedding', '_encoding', '_composition', '_classification']:
        gradients = [p.grad for p in getattr(model, name).parameters() if p.grad is not None]
        gradient_groups[name] = {'finite': all(bool(torch.isfinite(g).all()) for g in gradients),
                                 'absolute_sum': sum(g.abs().sum().item() for g in gradients)}
    nn.utils.clip_grad_norm_(model.parameters(), config['max_gradient_norm'])
    optimizer.step()
    changed = not torch.equal(before, model._classification[-1].weight.detach())
    if not changed or not all(row['finite'] and row['absolute_sum'] > 0 for row in gradient_groups.values()):
        raise RuntimeError('Upstream ESIM forward/backward/update smoke check failed.')
    report = dict(mode='one optimizer step only; not a training experiment', upstream=source,
                  torch_version=torch.__version__, config=config,
                  premise_shape=list(batch['premise'].shape), hypothesis_shape=list(batch['hypothesis'].shape),
                  logits_shape=list(logits.shape), labels_shape=list(batch['label'].shape), loss=loss.item(),
                  probabilities_sum_to_one=bool(torch.allclose(probabilities.sum(-1), torch.ones(len(logits), device=model.device))),
                  parameter_count=sum(p.numel() for p in model.parameters()), gradients=gradient_groups,
                  classifier_weights_changed=changed, labels={'0': 'entailment', '1': 'not_answer'})
    dump(output / 'smoke_report.json', report)
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/esim_debug.json')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    if args.epochs is not None:
        config['epochs'] = args.epochs
    if config['epochs'] < 1 or config['batch_size'] < 1 or not 0 < config['entailment_threshold'] < 1:
        raise ValueError('epochs/batch_size must be positive and threshold must be inside (0,1).')
    random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.set_num_threads(config['num_threads'])
    data = ROOT / config['data_dir']
    output = ROOT / (args.output if args.output else config['output_dir'] + ('_smoke' if args.smoke else ''))
    if any((output / name).exists() for name in ['manifest.json', 'smoke_report.json']) and not args.overwrite:
        raise FileExistsError(f'{output} already contains a run. Use --output or explicit --overwrite.')
    output.mkdir(parents=True, exist_ok=True)
    dump(output / 'config.json', config)
    source_hashes = {}
    for name in ['src/esim_adapter.py', 'src/phl.py', 'scripts/train_esim.py']:
        content = (ROOT / name).read_bytes()
        snapshot = output / 'source_snapshot' / name
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(content)
        source_hashes[name] = hashlib.sha256(content).hexdigest()
    vocabulary = json.loads((data / 'vocab.json').read_text(encoding='utf-8'))['token_to_id']
    data_manifest = json.loads((data / 'manifest.json').read_text(encoding='utf-8'))
    for name in ['train.jsonl', 'dev.jsonl', 'vocab.json', 'train.questions.jsonl', 'dev.questions.jsonl']:
        if digest(data / name) != data_manifest['outputs'][name]['sha256']:
            raise RuntimeError(f'{name} changed since PHL validation; regenerate or revalidate the data.')
    question_ids = {split: original_question_ids(data / f'{split}.questions.jsonl', config[f'{split}_question_limit'])
                    for split in ['train', 'dev']}
    datasets = {split: PHLDataset(data / f'{split}.jsonl', vocabulary,
                                 config['max_premise_tokens'], config['max_hypothesis_tokens'],
                                 allowed_question_ids=question_ids[split]) for split in ['train', 'dev']}
    if config['class_weights'] == 'balanced_train_only':
        counts = datasets['train'].label_counts
        if not all(counts):
            raise ValueError('Balanced binary loss requires both classes in the selected Train subset.')
        class_weights = torch.tensor([sum(counts) / (2 * count) for count in counts], dtype=torch.float, device=config['device'])
    elif config['class_weights'] is None:
        class_weights = torch.ones(2, device=config['device'])
    else:
        raise ValueError('class_weights must be balanced_train_only or null.')
    model, source = create_model(len(vocabulary), config)
    if args.smoke:
        smoke(model, datasets['train'], class_weights, config, output, source)
        return
    started = datetime.now(timezone.utc).isoformat()
    begin = time.perf_counter()
    loaders = {split: DataLoader(dataset, batch_size=config['batch_size'], shuffle=split == 'train',
                                num_workers=0, collate_fn=collate_phl,
                                generator=torch.Generator().manual_seed(config['seed']))
               for split, dataset in datasets.items()}
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    initial, _, _ = evaluate(model, loaders['dev'], class_weights, config['entailment_threshold'],
                              data / 'dev.questions.jsonl', question_ids['dev'])
    print('Initial Dev: ' + json.dumps(initial), flush=True)
    history = [dict(epoch=0, dev=initial)]
    best = None
    best_metrics = None
    for epoch in range(1, config['epochs'] + 1):
        epoch_begin = time.perf_counter()
        training = train_epoch(model, loaders['train'], optimizer, class_weights, config['max_gradient_norm'])
        validation, candidate_predictions, question_predictions = evaluate(model, loaders['dev'], class_weights,
            config['entailment_threshold'], data / 'dev.questions.jsonl', question_ids['dev'])
        history.append(dict(epoch=epoch, train=training, dev=validation, seconds=round(time.perf_counter() - epoch_begin, 3)))
        key = (validation['qa_exact_match'], -validation['loss'])
        checkpoint = dict(epoch=epoch, model=model.state_dict(), optimizer=optimizer.state_dict(), config=config,
                          upstream=source, vocabulary_sha256=digest(data / 'vocab.json'), dev_metrics=validation)
        torch.save(checkpoint, output / 'last.pt')
        if best is None or key > best:
            best = key
            best_metrics = dict(epoch=epoch, **validation)
            torch.save(checkpoint, output / 'best.pt')
            write_jsonl(output / 'dev_candidates.jsonl', candidate_predictions)
            write_jsonl(output / 'dev_questions.jsonl', question_predictions)
        dump(output / 'history.json', history)
        print(f'Epoch {epoch}: ' + json.dumps(history[-1]), flush=True)
    manifest = dict(started_utc=started, seconds=round(time.perf_counter() - begin, 3),
                    config=config, upstream=source, torch_version=torch.__version__,
                    parameter_count=sum(p.numel() for p in model.parameters()),
                    data_manifest_sha256=digest(data / 'manifest.json'),
                    adapter_source_sha256=source_hashes,
                    source_unchanged_during_run=all(digest(ROOT / name) == value for name, value in source_hashes.items()),
                    dataset={split: dict(questions=len(question_ids[split]), phl_rows=len(dataset),
                                         labels=dataset.label_counts, sequences_cropped=dataset.crop_counts)
                             for split, dataset in datasets.items()},
                    class_weights=class_weights.cpu().tolist(), initial_dev=initial, best_dev=best_metrics,
                    checkpoint_selection='highest Dev answer-set exact match, then lowest Dev weighted CE; threshold fixed by config',
                    no_test_data_used=True,
                    limitations=['Random trainable lexical embeddings, not Inference paper TransE/GloVe.',
                                 'Small dimensions and PHL debug subset; not full MetaQA/paper reproduction.',
                                 'Dev is used for model selection, not a final held-out Test result.'])
    dump(output / 'manifest.json', manifest)
    print('Completed: ' + str(output), flush=True)


if __name__ == '__main__':
    main()
