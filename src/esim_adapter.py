"""PHL adapters around the pinned, unmodified coetaur0/ESIM implementation."""
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import sys
import warnings

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from src.phl import tokenize

ROOT = Path(__file__).resolve().parents[1]


def upstream_source():
    lock = json.loads((ROOT / 'third_party/esim.lock.json').read_text(encoding='utf-8'))
    directory = ROOT / lock['directory']
    if not (directory / 'esim/model.py').is_file():
        raise FileNotFoundError('Pinned ESIM is missing. Run python -s -X utf8 scripts/setup_esim.py.')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=directory, text=True).strip()
    if commit != lock['commit']:
        raise RuntimeError('ESIM checkout differs from third_party/esim.lock.json.')
    if subprocess.check_output(['git', 'diff', '--name-only', 'HEAD'], cwd=directory, text=True).strip():
        raise RuntimeError('Upstream ESIM has local modifications; record them before using it.')
    sys.path.insert(0, str(directory))
    from esim.model import ESIM
    if not Path(inspect.getfile(ESIM)).resolve().is_relative_to(directory.resolve()):
        raise RuntimeError('A different installed esim package shadowed the pinned implementation.')
    source_hashes = {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                     for name in ['esim/model.py', 'esim/layers.py', 'esim/utils.py', 'LICENSE']}
    return ESIM, dict(repository=lock['repository'], commit=commit, license=lock['license'],
                      modified=False, source_sha256=source_hashes)


def create_model(vocabulary_size, config):
    if config['num_classes'] != 2:
        raise ValueError('The current PHL task has labels 0/1; configure num_classes=2.')
    if config['embedding_mode'] != 'random_trainable_debug':
        raise ValueError('Only explicit random-trainable debug embeddings are configured; add a verified pretrained mapping before changing this mode.')
    ESIM, source = upstream_source()
    warnings.filterwarnings('ignore', message='To copy construct from a tensor, it is recommended',
                            category=UserWarning, module=r'esim\.utils')
    model = ESIM(vocabulary_size, config['embedding_dim'], config['hidden_size'],
                 embeddings=None, padding_idx=0, dropout=config['dropout'],
                 num_classes=config['num_classes'], device=config['device']).to(config['device'])
    return model, source


class PHLDataset(Dataset):
    """Only encoded P/H enter the model; identity fields are for QA aggregation."""
    def __init__(self, path, vocabulary, max_premise=640, max_hypothesis=128, question_limit=None, allowed_question_ids=None):
        if min(max_premise, max_hypothesis) < 1 or (question_limit is not None and question_limit < 1):
            raise ValueError('Sequence/question limits must be positive, or question_limit=null.')
        self.rows, self.question_ids = [], set()
        self.crop_counts = {'premise': 0, 'hypothesis': 0}
        self.label_counts = [0, 0]
        with Path(path).open(encoding='utf-8') as stream:
            for line in stream:
                raw = json.loads(line)
                qid = raw['question_id']
                if allowed_question_ids is not None and qid not in allowed_question_ids:
                    continue
                if qid not in self.question_ids and question_limit is not None and len(self.question_ids) >= question_limit:
                    continue
                self.question_ids.add(qid)
                row = {key: raw[key] for key in ['sample_id', 'question_id', 'candidate', 'label']}
                for name, limit in [('premise', max_premise), ('hypothesis', max_hypothesis)]:
                    tokens = tokenize(raw[name])
                    if not tokens:
                        raise ValueError(f'{raw["sample_id"]}: empty {name} sequence')
                    self.crop_counts[name] += len(tokens) > limit
                    row[name] = [vocabulary.get(token, 1) for token in tokens[:limit]]
                if row['label'] not in (0, 1):
                    raise ValueError(f'{row["sample_id"]}: expected binary label')
                self.label_counts[row['label']] += 1
                self.rows.append(row)
        if not self.rows:
            raise ValueError(f'No PHL records selected from {path}')

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def collate_phl(rows):
    result = {}
    for name in ['premise', 'hypothesis']:
        sequences = [torch.tensor(row[name], dtype=torch.long) for row in rows]
        result[name] = pad_sequence(sequences, batch_first=True, padding_value=0)
        result[name + '_length'] = torch.tensor([len(row[name]) for row in rows], dtype=torch.long)
    result['label'] = torch.tensor([row['label'] for row in rows], dtype=torch.long)
    for name in ['sample_id', 'question_id', 'candidate']:
        result[name] = [row[name] for row in rows]
    return result


def model_inputs(batch, device):
    # pack_padded_sequence requires CPU lengths. IDs follow the configured device.
    return (batch['premise'].to(device), batch['premise_length'],
            batch['hypothesis'].to(device), batch['hypothesis_length'])


def qa_metrics(predictions, question_file, question_ids):
    """Exact answer sets, including empty predictions; denominator is questions."""
    selected = {}
    for row in predictions:
        if row['predicted_label'] == 0:
            selected.setdefault(row['question_id'], set()).add(row['candidate'])
    details, gold_total, candidate_hit, exact = [], 0, 0, 0
    with Path(question_file).open(encoding='utf-8') as stream:
        for line in stream:
            question = json.loads(line)
            qid = question['question_id']
            if qid not in question_ids:
                continue
            gold = set(question['answers'])
            predicted = selected.get(qid, set())
            hit = gold & set(question['candidates'])
            matched = predicted == gold
            exact += matched
            gold_total += len(gold)
            candidate_hit += len(hit)
            details.append(dict(question_id=qid, question=question['question'], answers=sorted(gold),
                                predicted_answers=sorted(predicted), exact_match=matched,
                                missing_answers=sorted(gold - predicted), extra_answers=sorted(predicted - gold)))
    if {row['question_id'] for row in details} != set(question_ids):
        raise ValueError('Question sidecar does not cover all evaluated question IDs.')
    return dict(questions=len(details), exact_correct=exact, qa_exact_match=exact / len(details),
                candidate_recall=candidate_hit / gold_total if gold_total else None), details
