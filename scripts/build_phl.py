"""Build and independently validate the teacher's real MetaQA PHL debug corpus.

Run from the repository: python -s -X utf8 scripts/build_phl.py
Only configured QA splits are read. No model, qtype sidecar or embeddings loaded.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.load_metaqa import load_triples, load_questions, parse_triple
from src.build_graph import build_graph
from src.phl import (candidate_entities, retrieve_evidence, select_training_candidates,
                     make_record, make_hypothesis, tokenize, make_batch)


def dump_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def write_row(stream, row):
    stream.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentiles(values):
    values = sorted(values)
    if not values:
        return dict(min=0, p50=0, p95=0, max=0)
    return dict(min=values[0], p50=values[len(values) // 2],
                p95=values[min(len(values) - 1, int(.95 * len(values)))], max=values[-1])


def validate_split(path, questions, config, raw_triples):
    """Re-read disk output: labels, raw facts, traversal, simple paths and text."""
    checked, seen_samples, covered = 0, set(), {}
    seen_candidates = {}
    for question in questions.values():
        covered[question['question_id']] = set()
        seen_candidates[question['question_id']] = set()
    with path.open(encoding='utf-8') as stream:
        for line_number, line in enumerate(stream, 1):
            row = json.loads(line)
            q = questions[row['question_id']]
            assert row['sample_id'] not in seen_samples, f'{path}:{line_number}: duplicate sample'
            seen_samples.add(row['sample_id'])
            assert row['candidate'] not in seen_candidates[q['question_id']]
            seen_candidates[q['question_id']].add(row['candidate'])
            assert row['split'] == q['split']
            assert row['question'] == q['question'] and row['topic_entity'] == q['topic_entity']
            is_positive = row['candidate'] in q['answers']
            expected = config['labels']['entailment'] if is_positive else config['labels']['not_answer']
            assert row['label'] == expected
            if is_positive:
                covered[q['question_id']].add(row['candidate'])
            assert row['paths'] and row['premise'] and row['hypothesis']
            assert row['hypothesis'] == make_hypothesis(q, row['candidate'])[0]
            fact_order, unique = [], set()
            for path_edges in row['paths']:
                assert 1 <= len(path_edges) <= config['max_hops']
                node = q['topic_entity']
                visited = {node}
                for edge in path_edges:
                    fact = tuple(edge['triple'])
                    assert raw_triples[edge['kb_line']] == fact
                    h, relation, t = fact
                    assert edge['relation'] == relation and isinstance(edge['inverse'], bool)
                    expected_source, expected_target = (t, h) if edge['inverse'] else (h, t)
                    assert edge['source'] == expected_source == node
                    assert edge['target'] == expected_target and expected_target not in visited
                    node = expected_target
                    visited.add(node)
                    if fact not in unique:
                        unique.add(fact)
                        fact_order.append(fact)
                assert node == row['candidate']
            assert row['premise_facts'] == [list(fact) for fact in fact_order]
            sentences = [config['relation_templates'][r].format(head=h, tail=t) for h, r, t in fact_order]
            assert row['premise_sentences'] == sentences
            assert row['premise'] == ' '.join(sentences)
            assert row['paths_available'] == row['paths_retained'] + row['paths_truncated']
            checked += 1
    missing = {qid: sorted(q['answers'] - covered[qid]) for qid, q in questions.items() if q['answers'] - covered[qid]}
    return dict(records_checked=checked, path_fact_label_text_checks='passed',
                questions_without_retained_gold=len(missing), missing_gold=missing)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/phl_debug.json')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--overwrite', action='store_true', help='Replace this generated output directory\'s named files')
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding='utf-8'))
    if config['labels'] != {'entailment': 0, 'not_answer': 1}:
        raise ValueError('This paper-compatible debug reader expects entailment=0, not_answer=1.')
    data = ROOT / config['dataset_dir']
    output = args.output.resolve() if args.output else ROOT / config['output_dir']
    if (output / 'manifest.json').exists() and not args.overwrite:
        raise FileExistsError(f'{output} already contains a completed run. Choose --output or use --overwrite explicitly.')
    output.mkdir(parents=True, exist_ok=True)
    begin = time.perf_counter()
    started = datetime.now(timezone.utc).isoformat()
    kb_path = data / 'kb.txt'
    triples = load_triples(kb_path)
    graph = build_graph(triples)
    raw_triples, triple_lines = {}, {}
    with kb_path.open(encoding='utf-8') as stream:
        for index, line in enumerate(stream, 1):
            if line.strip():
                fact = parse_triple(line)
                raw_triples[index] = fact
                triple_lines.setdefault(fact, index)
    source_files = {'kb': kb_path}
    stats, question_sets, previews = {}, {}, {}
    vocabulary_counts = Counter()
    dev_tokens = Counter()
    first_batch = []
    for split, limit in config['question_limits'].items():
        qa_path = data / f'2-hop/vanilla/qa_{split}.txt'
        source_files[split] = qa_path
        questions = load_questions(qa_path, split=split, limit=limit)
        question_sets[split] = {q['question_id']: q for q in questions}
        labels, rules, lengths, path_lengths = Counter(), Counter(), [], []
        total_gold = candidate_gold = total_candidates = rows_written = path_count = truncated = 0
        full_gold_questions = 0
        dataset_part = output / f'{split}.jsonl.partial'
        question_part = output / f'{split}.questions.jsonl.partial'
        with dataset_part.open('w', encoding='utf-8', newline='\n') as samples, question_part.open('w', encoding='utf-8', newline='\n') as qout:
            for number, q in enumerate(questions, 1):
                pool = candidate_entities(graph, q['topic_entity'],
                                          set(config['candidate_first_hop_relations']), config['candidate_hops'])
                hit = pool & q['answers']
                total_gold += len(q['answers'])
                candidate_gold += len(hit)
                full_gold_questions += hit == q['answers']
                total_candidates += len(pool)
                selected = (select_training_candidates(pool, q['answers'], config['negative_per_positive'],
                                                        config['seed'], q['question_id']) if split == 'train' else sorted(pool))
                evidence, available = retrieve_evidence(graph, q['topic_entity'], set(selected),
                                                        config['max_hops'], config['max_paths_per_candidate'])
                retained = []
                for candidate in selected:
                    paths = evidence.get(candidate, [])
                    if not paths:
                        continue
                    row = make_record(q, candidate, paths, available[candidate], config, triple_lines)
                    write_row(samples, row)
                    retained.append(candidate)
                    rows_written += 1
                    labels[str(row['label'])] += 1
                    rules[row['hypothesis_rule']] += 1
                    tokens = tokenize(row['premise']) + tokenize(row['hypothesis'])
                    if split == 'train':
                        vocabulary_counts.update(tokens)
                        if len(first_batch) < 8:
                            first_batch.append(row)
                    else:
                        dev_tokens.update(tokens)
                    lengths.append(len(tokenize(row['premise'])))
                    path_lengths.append(len(paths))
                    path_count += len(paths)
                    truncated += row['paths_truncated']
                    preview_key = f'{split}_label_{row["label"]}'
                    if preview_key not in previews:
                        previews[preview_key] = row
                    if split == 'train' and len(paths) > 1 and 'train_multiple_paths' not in previews:
                        previews['train_multiple_paths'] = row
                write_row(qout, dict(question_id=q['question_id'], split=split, question=q['question'],
                                     topic_entity=q['topic_entity'], answers=sorted(q['answers']),
                                     candidates=sorted(pool), retained_candidates=retained,
                                     missing_gold_candidates=sorted(q['answers'] - pool),
                                     missing_gold_evidence=sorted(q['answers'] - set(retained)),
                                     source_line=int(q['question_id'].split(':')[-1])))
                if number % 100 == 0:
                    print(f'{split}: {number}/{len(questions)} questions, {rows_written} PHL rows', flush=True)
        final_path = output / f'{split}.jsonl'
        dataset_part.replace(final_path)
        question_part.replace(output / f'{split}.questions.jsonl')
        stats[split] = dict(questions=len(questions), phl_samples=rows_written, labels=dict(labels),
                            candidate_pairs=total_candidates, gold_answers=total_gold,
                            gold_answers_retrieved=candidate_gold,
                            candidate_recall=candidate_gold / total_gold if total_gold else None,
                            questions_with_all_gold_candidates=full_gold_questions,
                            stored_paths=path_count, truncated_paths=truncated,
                            premise_token_lengths=percentiles(lengths), paths_per_sample=percentiles(path_lengths),
                            samples_over_batch_premise_limit=sum(n > config['batch_max_premise_tokens'] for n in lengths),
                            hypothesis_rules=dict(sorted(rules.items())))
    vocabulary = {'<pad>': 0, '<unk>': 1}
    for token, _count in sorted(vocabulary_counts.items(), key=lambda pair: (-pair[1], pair[0])):
        if token not in vocabulary:
            vocabulary[token] = len(vocabulary)
    dump_json(output / 'vocab.json', dict(token_to_id=vocabulary, pad_id=0, unk_id=1,
                                        scope='train P/H text only; case-sensitive lexical tokens; no pretrained vectors'))
    batch = make_batch(first_batch, vocabulary, config['batch_max_premise_tokens'], config['batch_max_hypothesis_tokens'])
    dump_json(output / 'batch_example.json', batch)
    dump_json(output / 'preview.json', previews)
    validation = {split: validate_split(output / f'{split}.jsonl', questions, config, raw_triples)
                  for split, questions in question_sets.items()}
    train_text = {q['question'] for q in question_sets.get('train', {}).values()}
    dev_text = {q['question'] for q in question_sets.get('dev', {}).values()}
    validation['exact_train_dev_question_overlap'] = len(train_text & dev_text)
    validation['vocabulary_scope'] = 'train only'
    validation['dev_oov_token_rate'] = (sum(n for t, n in dev_tokens.items() if t not in vocabulary) / sum(dev_tokens.values())) if dev_tokens else None
    validation['batch_example'] = dict(size=len(first_batch),
                                     premise_shape=[len(first_batch), len(batch['premise_ids'][0]) if first_batch else 0],
                                     hypothesis_shape=[len(first_batch), len(batch['hypothesis_ids'][0]) if first_batch else 0],
                                     labels_shape=[len(first_batch)],
                                     premise_cropped=batch['premise_cropped'], hypothesis_cropped=batch['hypothesis_cropped'])
    for name in ['premise', 'hypothesis']:
        assert len({len(row) for row in batch[name + '_ids']}) <= 1
        assert all(len(ids) == len(mask) for ids, mask in zip(batch[name + '_ids'], batch[name + '_mask']))
    try:
        git_head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        git_head = None
    dump_json(output / 'validation.json', validation)
    generated = [p for p in output.iterdir() if p.suffix in ('.json', '.jsonl') and p.name != 'manifest.json']
    source_code = [ROOT / p for p in ['src/phl.py', 'src/load_metaqa.py', 'src/build_graph.py', 'scripts/build_phl.py']]
    manifest = dict(schema_version=1, started_utc=started, elapsed_seconds=round(time.perf_counter() - begin, 3),
                    python=platform.python_version(), git_base_commit=git_head, config=config,
                    source_code_sha256={str(p.relative_to(ROOT)): sha256(p) for p in source_code},
                    config_sha256=sha256(config_path),
                    data_sources={key: dict(path=str(path.relative_to(ROOT)), sha256=sha256(path)) for key, path in source_files.items()},
                    selection='first N questions per official split; original line IDs retained',
                    candidate_protocol='exact two-hop endpoints; first edge movie-person relation; no answers/qtype input',
                    evidence_protocol='all 1/2-hop node-simple paths in full KB to selected candidates; all relations; deduplicated routes',
                    premise_protocol='one English template sentence per unique raw triple, preserving first occurrence order; all raw paths retained',
                    hypothesis_protocol='21 vanilla semantic forms inferred from question text; not qtype sidecars or test gold chains',
                    vocabulary_size=len(vocabulary), stats=stats, validation=validation,
                    outputs={p.name: dict(bytes=p.stat().st_size, sha256=sha256(p)) for p in generated},
                    limitations=['Teacher reference/debug subset; not the author-released PHL corpus or a full paper reproduction.',
                                 'Candidate schema and English declarative templates are documented implementation choices.',
                                 'Raw PHL text is not truncated; collation may crop sequences and reports the count.',
                                 'KGE/GloVe resources and ESIM/HRPE model training have not been completed.'])
    dump_json(output / 'manifest.json', manifest)
    report = ['# MetaQA-2hop PHL debug corpus', '', 'Generated from the local official vanilla files. Teacher reference, not a course submission.', '',
              '| Split | Questions | PHL rows | Label 0 | Label 1 | Gold candidate recall |', '|---|---:|---:|---:|---:|---:|']
    for split, info in stats.items():
        report.append(f'| {split} | {info["questions"]} | {info["phl_samples"]} | {info["labels"].get("0", 0)} | {info["labels"].get("1", 0)} | {info["candidate_recall"]:.2%} |')
    report += ['', 'Model input: `premise` + `hypothesis`; supervised target: `label` (0 supports, 1 not an answer).',
               'Do not feed answers, source qtype, or other gold-only metadata into the model.',
               'Question-level metadata and full candidate lists: `*.questions.jsonl`; all raw paths and KB line IDs are in PHL rows.',
               'Candidate first-hop schema and English templates are adaptations; see `manifest.json` for the exact protocol.',
               'All candidates are retained by default; the Dev denominator is the selected original questions.',
               'No Test file was read. Vocabulary uses Train only. No model training or pretrained KG/GloVe preparation was performed.',
               'See `preview.json`, `batch_example.json`, `vocab.json`, and `validation.json` for inspection and collation.',
               'Premises remain untruncated in JSONL; `make_batch` applies configurable sequence limits and returns crop counters.']
    (output / 'README.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
    print(json.dumps({'output': str(output), 'stats': stats, 'validation': validation,
                      'seconds': manifest['elapsed_seconds']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
