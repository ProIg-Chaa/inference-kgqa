"""Small correctness checks that run without downloading the dataset."""
import unittest

from src.load_metaqa import parse_question, parse_triple
from src.build_graph import build_graph
from src.path_retrieval import find_paths, path_nodes


class Lesson02Tests(unittest.TestCase):
    def test_question_preserves_multiple_answers_and_split(self):
        sample = parse_question('who acted in [Film]?\tActor A|Actor B\n', 'train:7', 'train')
        self.assertEqual(sample['answers'], {'Actor A', 'Actor B'})
        self.assertEqual(sample['topic_entity'], 'Film')
        self.assertEqual((sample['question_id'], sample['split']), ('train:7', 'train'))

    def test_malformed_input_is_reported(self):
        with self.assertRaises(ValueError):
            parse_triple('A|r')
        with self.assertRaises(ValueError):
            parse_question('no entity marked\tB', 'train:1', 'train')

    def test_parallel_relations_are_not_overwritten(self):
        graph = build_graph([('Film', 'directed_by', 'Person'), ('Film', 'written_by', 'Person')])
        paths = find_paths(graph, 'Film', 'Person', max_hops=1)
        self.assertEqual({p[0]['relation'] for p in paths}, {'directed_by', 'written_by'})

    def test_inverse_access_keeps_original_fact(self):
        graph = build_graph([('Film', 'directed_by', 'Person')])
        step = find_paths(graph, 'Person', 'Film', max_hops=1)[0][0]
        self.assertTrue(step['inverse'])
        self.assertEqual(step['triple'], ('Film', 'directed_by', 'Person'))

    def test_diamond_routes_survive_and_cycles_are_excluded(self):
        graph = build_graph([('A', 'r', 'B'), ('A', 's', 'C'),
                             ('B', 't', 'D'), ('C', 'u', 'D'), ('B', 'v', 'C')])
        paths = find_paths(graph, 'A', 'D', max_hops=2)
        self.assertEqual({tuple(path_nodes('A', p)) for p in paths}, {('A', 'B', 'D'), ('A', 'C', 'D')})
        for path in find_paths(graph, 'A', max_hops=3):
            nodes = path_nodes('A', path)
            self.assertEqual(len(nodes), len(set(nodes)))

    def test_bound_and_unreachable_target(self):
        graph = build_graph([('A', 'r', 'B'), ('B', 's', 'C')])
        self.assertEqual(find_paths(graph, 'A', 'C', max_hops=1), [])
        self.assertEqual(find_paths(graph, 'A', 'missing'), [])
        self.assertEqual(find_paths(graph, 'A', 'A', max_hops=0), [[]])

    def test_optional_output_cap(self):
        graph = build_graph([('A', 'r', 'B'), ('A', 's', 'C')])
        self.assertEqual(len(find_paths(graph, 'A', max_hops=1, max_paths=1)), 1)
        with self.assertRaises(ValueError):
            find_paths(graph, 'A', max_hops=-1)


if __name__ == '__main__':
    unittest.main()
