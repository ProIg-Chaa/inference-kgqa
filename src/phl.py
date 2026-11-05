"""Auditable MetaQA-2hop -> PHL teacher reference; no neural model dependency.

Question templates are inferred from question text, never qa_*_qtype.txt.
Candidate retrieval never receives answers. Labels use the supplied QA answers
after retrieval; class 1 means a benchmark negative, not strict contradiction.
"""
from collections import defaultdict
import hashlib
import math
import random
import re


def hypothesis_rule(question):
    """Recognize supported vanilla templates; fail rather than invent a fallback."""
    text = re.sub(r'\[[^]]+\]', '[entity]', question).lower().strip(' ?')
    if re.search(r'co-star|co-stars|acted together|starred together|appeared in the same movie', text):
        return 'actor_to_movie_to_actor'
    if re.search(r'co-direct|directed (?:films|movies) together', text):
        return 'director_to_movie_to_director'
    if re.search(r'co-wr|wrote (?:films|movies) together', text):
        return 'writer_to_movie_to_writer'
    movie_rules = [
        ('director', r'same director|the director of \[entity\]'),
        ('actor', r'same actor|the actor (?:of|in) \[entity\]'),
        ('writer', r'same screenwriter|share the screenwriter|the (?:screenwriter|scriptwriter) of \[entity\]'),
    ]
    for middle, pattern in movie_rules:
        if re.search(pattern, text):
            return f'movie_to_{middle}_to_movie'
    source_patterns = [
        ('director', [r'directed by \[entity\]', r'\[entity\] directed', r'the director \[entity\]']),
        ('writer', [r'written by \[entity\]', r'\[entity\] written', r'(?:screenwriter|writer) \[entity\]']),
        ('actor', [r'(?:acted|starred) by \[entity\]', r'\[entity\] (?:acted|starred)']),
    ]
    for source, patterns in source_patterns:
        for pattern in patterns:
            match = re.search(pattern, text)
            if not match:
                continue
            remaining = text[:match.start()] + ' [source] ' + text[match.end():]
            targets = [
                ('year', r'\bwhen\b|release|\byears?\b'),
                ('language', r'languages?'),
                ('genre', r'genres?|\btypes?\b'),
                ('director', r'director|directed'),
                ('writer', r'wrote|written|writers?|screenwriter|screenplay'),
                ('actor', r'actors?|starred|acted'),
            ]
            for target, target_pattern in targets:
                if re.search(target_pattern, remaining):
                    return f'{source}_to_movie_to_{target}'
    raise ValueError(f'Unsupported MetaQA vanilla question template: {question!r}')


def make_hypothesis(question, candidate):
    """Declarative, existential paraphrases of supported two-hop questions."""
    rule = hypothesis_rule(question['question'])
    source, middle, target = rule.split('_to_')
    topic = question['topic_entity']
    if source == 'movie':
        noun = {'director': 'a director', 'writer': 'a screenwriter', 'actor': 'an actor'}[middle]
        return f'{candidate} shares {noun} with {topic}.', rule
    if source == target:
        verb = {'actor': 'acted in a movie', 'writer': 'co-wrote a movie', 'director': 'co-directed a movie'}[source]
        return f'{candidate} {verb} with {topic}.', rule
    clause = {'actor': f'starring {topic}', 'writer': f'written by {topic}', 'director': f'directed by {topic}'}[source]
    patterns = {
        'actor': '{candidate} acted in a movie {clause}.',
        'writer': '{candidate} wrote a movie {clause}.',
        'director': '{candidate} directed a movie {clause}.',
        'genre': 'A movie {clause} has the genre {candidate}.',
        'language': 'A movie {clause} is in {candidate}.',
        'year': 'A movie {clause} was released in {candidate}.',
    }
    return patterns[target].format(candidate=candidate, clause=clause), rule


def candidate_entities(graph, start, first_hop_relations, hops=2):
    """Schema-bounded candidates: first edge must be a movie-person relation.

    MetaQA-2hop's supported 21 templates traverse a movie/person link first.
    This documented adaptation avoids exploding via genre/year/tag hubs.
    It is independent of answers and question-type sidecar files.
    """
    if hops != 2:
        raise ValueError('This candidate protocol currently supports exactly 2 hops.')
    candidates = set()
    for first in graph.get(start, []):
        middle = first['target']
        if first['relation'] not in first_hop_relations or middle == start:
            continue
        for second in graph.get(middle, []):
            end = second['target']
            if end not in (start, middle):
                candidates.add(end)
    return candidates


def retrieve_evidence(graph, start, candidates, max_hops=2, max_paths=None):
    """All 1/2-hop simple evidence routes to candidates, over ALL KG relations.

    Evidence is broader than the schema-bounded candidate-generation routes.
    A cap, if requested, is per candidate and its truncation count is returned.
    """
    if max_hops not in (1, 2):
        raise ValueError('The debug evidence enumerator supports max_hops 1 or 2.')
    if max_paths is not None and max_paths < 1:
        raise ValueError('max_paths_per_candidate must be positive or null.')
    grouped, counts, seen = defaultdict(list), defaultdict(int), defaultdict(set)

    def save(path):
        end = path[-1]['target']
        if end not in candidates:
            return
        key = tuple((edge['source'], edge['relation'], edge['inverse'], edge['target']) for edge in path)
        if key in seen[end]:
            return
        seen[end].add(key)
        counts[end] += 1
        if max_paths is None or len(grouped[end]) < max_paths:
            grouped[end].append(path)

    for first in graph.get(start, []):
        middle = first['target']
        if middle == start:
            continue
        first_step = dict(source=start, **first)
        save([first_step])
        if max_hops == 1:
            continue
        for second in graph.get(middle, []):
            if second['target'] in (start, middle):
                continue
            save([first_step, dict(source=middle, **second)])
    return dict(grouped), dict(counts)


def select_training_candidates(candidates, answers, ratio, seed, question_id):
    """Optional negative sampling after retrieval; default keeps every candidate."""
    ordered = sorted(candidates)
    if ratio is None:
        return ordered
    if ratio < 0:
        raise ValueError('negative_per_positive must be nonnegative or null.')
    positives = sorted(candidates & answers)
    negatives = sorted(candidates - answers)
    requested = min(len(negatives), math.ceil(ratio * max(1, len(positives))))
    key = hashlib.sha256(f'{seed}:{question_id}'.encode('utf-8')).digest()
    rng = random.Random(int.from_bytes(key, 'big'))
    return sorted(positives + rng.sample(negatives, requested))


def make_record(question, candidate, paths, path_count, config, triple_lines):
    if not paths:
        raise ValueError('Cannot create a training sample without evidence paths.')
    hypothesis, rule = make_hypothesis(question, candidate)
    sentences, unique_facts, seen = [], [], set()
    serialized_paths = []
    for path in paths:
        serialized = []
        for edge in path:
            triple = tuple(edge['triple'])
            head, relation, tail = triple
            if relation not in config['relation_templates']:
                raise ValueError(f'Missing text template for relation {relation!r}')
            if triple not in seen:
                seen.add(triple)
                unique_facts.append(list(triple))
                sentences.append(config['relation_templates'][relation].format(head=head, tail=tail))
            serialized.append(dict(source=edge['source'], target=edge['target'],
                                   relation=relation, inverse=edge['inverse'], triple=list(triple),
                                   kb_line=triple_lines[triple]))
        serialized_paths.append(serialized)
    suffix = hashlib.sha256(candidate.encode('utf-8')).hexdigest()[:16]
    return dict(schema_version=1, sample_id=f"{question['question_id']}:{suffix}",
                question_id=question['question_id'], split=question['split'],
                question=question['question'], topic_entity=question['topic_entity'], candidate=candidate,
                premise=' '.join(sentences), premise_sentences=sentences,
                premise_facts=unique_facts, hypothesis=hypothesis, hypothesis_rule=rule,
                label=config['labels']['entailment'] if candidate in question['answers'] else config['labels']['not_answer'],
                paths=serialized_paths, paths_available=path_count,
                paths_retained=len(paths), paths_truncated=path_count - len(paths))


def tokenize(text):
    """Case-sensitive lexical tokens; preserve raw KG IDs separately in paths."""
    return re.findall(r"\w+(?:['’-]\w+)*|[^\w\s]", text, flags=re.UNICODE)


def make_batch(records, vocabulary, max_premise=512, max_hypothesis=128):
    """Standard-library collation; lists can be converted directly to tensors.

    Cropping happens here, not silently in the stored PHL records. Counters expose
    how many sequences exceed the requested batch limit. KGE/GloVe are separate.
    """
    if max_premise < 1 or max_hypothesis < 1:
        raise ValueError('Batch sequence limits must be positive.')
    batch = {'labels': [row['label'] for row in records], 'sample_ids': [row['sample_id'] for row in records]}
    for name, limit in [('premise', max_premise), ('hypothesis', max_hypothesis)]:
        encoded, cropped = [], 0
        for row in records:
            tokens = tokenize(row[name])
            cropped += len(tokens) > limit
            encoded.append([vocabulary.get(token, 1) for token in tokens[:limit]])
        width = max(map(len, encoded), default=0)
        batch[name + '_ids'] = [ids + [0] * (width - len(ids)) for ids in encoded]
        batch[name + '_mask'] = [[1] * len(ids) + [0] * (width - len(ids)) for ids in encoded]
        batch[name + '_cropped'] = cropped
    return batch
