"""Lesson 02 teacher reference: read original MetaQA text without changing labels."""
from pathlib import Path
import re


def parse_triple(line):
    fields = line.rstrip('\r\n').split('|')
    if len(fields) != 3 or not all(fields):
        raise ValueError('A KG line must contain nonempty subject|relation|object')
    return tuple(fields)


def parse_question(line, question_id, split):
    fields = line.rstrip('\r\n').split('\t')
    if len(fields) != 2 or not all(fields):
        raise ValueError('A QA line must contain question<TAB>answer(s)')
    question, answer_text = fields
    entities = re.findall(r'\[([^\]]+)\]', question)
    if len(entities) != 1:
        raise ValueError('This lesson expects one bracketed topic entity')
    answers = set(answer_text.split('|'))
    if '' in answers:
        raise ValueError('An answer entity is empty')
    return {'question_id': question_id, 'split': split, 'question': question,
            'topic_entity': entities[0], 'answers': answers}


def load_triples(path):
    triples = []
    with Path(path).open(encoding='utf-8') as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                triples.append(parse_triple(line))
            except ValueError as error:
                raise ValueError(f'{path}:{line_number}: {error}') from error
    return triples


def load_questions(path, split='train', limit=None):
    if limit is not None and limit < 0:
        raise ValueError('limit must be nonnegative or None')
    questions = []
    with Path(path).open(encoding='utf-8') as stream:
        for line_number, line in enumerate(stream, 1):
            if limit is not None and len(questions) >= limit:
                break
            if not line.strip():
                continue
            try:
                questions.append(parse_question(line, f'{split}:{line_number}', split))
            except ValueError as error:
                raise ValueError(f'{path}:{line_number}: {error}') from error
    return questions
