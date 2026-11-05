"""Reload a local ESIM checkpoint and re-evaluate the recorded independent Dev."""
import argparse
import json
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.esim_adapter import PHLDataset, collate_phl, create_model
from train_esim import evaluate, dump, digest, original_question_ids, write_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    checkpoint_path = args.checkpoint.resolve()
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    run_manifest = json.loads((checkpoint_path.parent / 'manifest.json').read_text(encoding='utf-8'))
    config = checkpoint['config']
    torch.set_num_threads(config['num_threads'])
    data = ROOT / config['data_dir']
    if digest(data / 'vocab.json') != checkpoint['vocabulary_sha256']:
        raise ValueError('Checkpoint vocabulary differs from the current vocabulary file.')
    if digest(data / 'manifest.json') != run_manifest['data_manifest_sha256']:
        raise ValueError('PHL data manifest differs from the training run.')
    vocabulary = json.loads((data / 'vocab.json').read_text(encoding='utf-8'))['token_to_id']
    question_ids = original_question_ids(data / 'dev.questions.jsonl', config['dev_question_limit'])
    dataset = PHLDataset(data / 'dev.jsonl', vocabulary, config['max_premise_tokens'],
                         config['max_hypothesis_tokens'], allowed_question_ids=question_ids)
    model, source = create_model(len(vocabulary), config)
    if source != checkpoint['upstream']:
        raise ValueError('Upstream ESIM source differs from the checkpoint provenance.')
    model.load_state_dict(checkpoint['model'])
    loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_phl)
    weights = torch.tensor(run_manifest['class_weights'], dtype=torch.float, device=config['device'])
    metrics, candidates, questions = evaluate(model, loader, weights, config['entailment_threshold'],
                                              data / 'dev.questions.jsonl', question_ids)
    output = args.output.resolve() if args.output else checkpoint_path.parent / 'evaluation_dev'
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / 'candidates.jsonl', candidates)
    write_jsonl(output / 'questions.jsonl', questions)
    recorded = checkpoint['dev_metrics']
    agreement = all(metrics[name] == recorded[name] for name in ['exact_correct', 'qa_exact_match', 'candidate_accuracy', 'confusion_matrix_gold_rows'])
    if not agreement:
        raise RuntimeError('Reloaded Dev predictions differ from the checkpoint metrics; inspect configuration/data/device.')
    report = dict(checkpoint=str(checkpoint_path), checkpoint_sha256=digest(checkpoint_path),
                  upstream=source, checkpoint_epoch=checkpoint['epoch'], metrics=metrics,
                  recorded_metrics_match=agreement, evaluation_source_sha256=digest(Path(__file__)))
    dump(output / 'report.json', report)
    print(json.dumps(report))


if __name__ == '__main__':
    main()
