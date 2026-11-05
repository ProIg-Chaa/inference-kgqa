"""Semantic/data-boundary checks for PHL; no actual dataset download needed."""
import unittest

from src.build_graph import build_graph
from src.phl import (hypothesis_rule, make_hypothesis, candidate_entities,
                     retrieve_evidence, select_training_candidates, make_record, make_batch)


class PHLTests(unittest.TestCase):
    def test_template_semantics_cover_21_structures_without_qtype_input(self):
        for source, phrase in [('actor', 'acted by'), ('writer', 'written by'), ('director', 'directed by')]:
            target_questions = {
                'actor': f'who acted in the movies {phrase} [Person]',
                'writer': f'who wrote the movies {phrase} [Person]',
                'director': f'who directed the films {phrase} [Person]',
                'genre': f'what genres are the films {phrase} [Person] in',
                'language': f'what languages are the films {phrase} [Person] in',
                'year': f'when were the movies {phrase} [Person] released',
            }
            for target, question in target_questions.items():
                self.assertEqual(hypothesis_rule(question), f'{source}_to_movie_to_{target}')
        for middle, question in [
            ('director', 'which movies have the same director of [Film]'),
            ('writer', 'the scriptwriter of [Film] also wrote movies'),
            ('actor', 'the actor in [Film] also appears in which movies'),
        ]:
            self.assertEqual(hypothesis_rule(question), f'movie_to_{middle}_to_movie')
        with self.assertRaises(ValueError):
            hypothesis_rule('what is the budget of [Film]')

    def test_candidates_do_not_inject_gold_and_skip_attribute_hubs(self):
        triples = [('Film A', 'directed_by', 'Director'), ('Film B', 'directed_by', 'Director'),
                   ('Film A', 'has_genre', 'Drama'), ('Unrelated Film', 'has_genre', 'Drama')]
        graph = build_graph(triples)
        candidates = candidate_entities(graph, 'Film A', {'directed_by'})
        self.assertEqual(candidates, {'Film B'})
        self.assertNotIn('Never Retrieved Gold', select_training_candidates(candidates, {'Never Retrieved Gold'}, None, 7, 'train:1'))

    def test_evidence_still_keeps_other_relations_and_caps_per_candidate(self):
        graph = build_graph([('Film A', 'directed_by', 'Director'), ('Film B', 'directed_by', 'Director'),
                             ('Film A', 'has_genre', 'Drama'), ('Film B', 'has_genre', 'Drama')])
        evidence, counts = retrieve_evidence(graph, 'Film A', {'Film B'})
        self.assertEqual(len(evidence['Film B']), 2)
        self.assertEqual(counts['Film B'], 2)
        capped, counts = retrieve_evidence(graph, 'Film A', {'Film B'}, max_paths=1)
        self.assertEqual((len(capped['Film B']), counts['Film B']), (1, 2))

    def test_negative_sampling_is_deterministic_and_retains_all_positives(self):
        pool, gold = {'a', 'b', 'c', 'd', 'e'}, {'a', 'c'}
        selected = select_training_candidates(pool, gold, .5, 7, 'train:7')
        self.assertEqual(selected, select_training_candidates(pool, gold, .5, 7, 'train:7'))
        self.assertTrue(gold <= set(selected))
        self.assertEqual(len(selected), 3)

    def test_reverse_evidence_text_and_labels_follow_original_facts(self):
        triples = [('Movie', 'directed_by', 'Director'), ('Movie', 'written_by', 'Writer'), ('Movie', 'starred_actors', 'Actor')]
        graph = build_graph(triples)
        q = dict(question='who wrote movies directed by [Director]', topic_entity='Director',
                 question_id='train:1', split='train', answers={'Writer'})
        config = {'labels': {'entailment': 0, 'not_answer': 1},
                  'relation_templates': {'directed_by': '{head} is directed by {tail}.',
                                         'written_by': '{head} is written by {tail}.',
                                         'starred_actors': '{head} stars {tail}.'}}
        paths, counts = retrieve_evidence(graph, 'Director', {'Writer', 'Actor'})
        line_ids = {triple: index for index, triple in enumerate(triples, 1)}
        positive = make_record(q, 'Writer', paths['Writer'], counts['Writer'], config, line_ids)
        negative = make_record(q, 'Actor', paths['Actor'], counts['Actor'], config, line_ids)
        self.assertEqual((positive['label'], negative['label']), (0, 1))
        self.assertIn('Movie is directed by Director.', positive['premise'])
        self.assertNotIn('Director is directed by Movie.', positive['premise'])
        self.assertIn('wrote a movie', negative['hypothesis'])
        self.assertTrue(positive['paths'][0][0]['inverse'])

    def test_batch_mask_oov_and_visible_cropping(self):
        records = [dict(sample_id='a', premise='Known unseen tail', hypothesis='Known', label=0),
                   dict(sample_id='b', premise='Known', hypothesis='Known', label=1)]
        batch = make_batch(records, {'<pad>': 0, '<unk>': 1, 'Known': 2}, max_premise=2)
        self.assertEqual(batch['premise_ids'], [[2, 1], [2, 0]])
        self.assertEqual(batch['premise_mask'], [[1, 1], [1, 0]])
        self.assertEqual(batch['premise_cropped'], 1)


if __name__ == '__main__':
    unittest.main()
