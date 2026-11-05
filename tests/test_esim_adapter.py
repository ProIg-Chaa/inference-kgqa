"""Optional isolated-PyTorch checks: padding/sorting and question-level evaluation."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

HAS_TORCH = importlib.util.find_spec('torch') is not None
HAS_UPSTREAM = (Path(__file__).resolve().parents[1] / 'third_party/ESIM/esim/model.py').is_file()


@unittest.skipUnless(HAS_TORCH and HAS_UPSTREAM, 'Run in the isolated Torch environment after setup_esim.py')
class ESIMAdapterTests(unittest.TestCase):
    def test_padding_sorting_and_single_batch_agree(self):
        import torch
        from src.esim_adapter import create_model, collate_phl, model_inputs
        torch.set_num_threads(1)
        torch.manual_seed(7)
        model, _ = create_model(8, dict(num_classes=2, embedding_mode='random_trainable_debug',
                                       embedding_dim=8, hidden_size=4, dropout=0.0, device='cpu'))
        model.eval()
        rows = [dict(sample_id='a', question_id='dev:1', candidate='A', label=0, premise=[2], hypothesis=[3, 4]),
                dict(sample_id='b', question_id='dev:1', candidate='B', label=1, premise=[2, 5, 6], hypothesis=[3])]
        batch = collate_phl(rows)
        with torch.no_grad():
            logits, probabilities = model(*model_inputs(batch, 'cpu'))
            self.assertEqual(list(logits.shape), [2, 2])
            self.assertTrue(torch.isfinite(logits).all())
            self.assertTrue(torch.allclose(probabilities.sum(-1), torch.ones(2)))
            for index, row in enumerate(rows):
                single, _ = model(*model_inputs(collate_phl([row]), 'cpu'))
                self.assertTrue(torch.allclose(logits[index], single[0], atol=1e-6))
            padded = dict(batch)
            padded['premise'] = torch.nn.functional.pad(batch['premise'], (0, 3))
            padded_logits, _ = model(*model_inputs(padded, 'cpu'))
            self.assertTrue(torch.allclose(logits, padded_logits, atol=1e-6))

    def test_qa_sets_include_empty_predictions_and_original_denominator(self):
        from src.esim_adapter import qa_metrics
        questions = [dict(question_id='dev:1', question='Q1', answers=['A', 'B'], candidates=['A', 'B', 'C']),
                     dict(question_id='dev:2', question='Q2', answers=['D'], candidates=[])]
        def predictions(names):
            return [dict(question_id='dev:1', candidate=name, predicted_label=0) for name in names]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'questions.jsonl'
            path.write_text('\n'.join(json.dumps(q) for q in questions), encoding='utf-8')
            partial, _ = qa_metrics(predictions(['A']), path, {'dev:1', 'dev:2'})
            extra, _ = qa_metrics(predictions(['A', 'B', 'C']), path, {'dev:1', 'dev:2'})
            correct, details = qa_metrics(predictions(['A', 'B']), path, {'dev:1', 'dev:2'})
            self.assertEqual((partial['qa_exact_match'], extra['qa_exact_match']), (0, 0))
            self.assertEqual((correct['questions'], correct['exact_correct'], correct['qa_exact_match']), (2, 1, .5))
            self.assertEqual(details[1]['predicted_answers'], [])


if __name__ == '__main__':
    unittest.main()
