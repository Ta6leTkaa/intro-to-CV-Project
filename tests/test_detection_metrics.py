"""Checks for metric errors that would change threshold selection."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from detection_metrics import evaluate_threshold, match_detections


class DetectionMetricsTests(unittest.TestCase):
    def evaluate(self, predictions, targets, threshold=0.25):
        return evaluate_threshold([dict(predictions=predictions, targets=targets)], threshold, ['a', 'b'])

    def test_perfect_detection_and_absent_class(self):
        r = self.evaluate([[0, 0, 10, 10, 0.8, 0]], [[0, 0, 10, 10, 0]])
        self.assertEqual(r['macro_f1'], 1.0)
        self.assertIsNone(r['per_class'][1]['recall'])
        self.assertIsNone(r['per_class'][1]['f1'])

    def test_duplicate_predictions_do_not_reuse_target(self):
        r = self.evaluate([[0, 0, 10, 10, 0.8, 0], [0, 0, 10, 10, 0.7, 0]], [[0, 0, 10, 10, 0]])
        self.assertEqual((r['per_class'][0]['tp'], r['per_class'][0]['fp']), (1, 1))
        self.assertAlmostEqual(r['macro_f1'], 2 / 3)

    def test_wrong_class_is_not_true_positive(self):
        r = self.evaluate([[0, 0, 10, 10, 0.8, 1]], [[0, 0, 10, 10, 0]])
        self.assertEqual(r['macro_f1'], 0)
        self.assertEqual(r['absent_class_false_positives'], 1)

    def test_no_predictions_and_no_ground_truth(self):
        self.assertIsNone(self.evaluate([], [])['macro_f1'])
        self.assertEqual(self.evaluate([], [[0, 0, 10, 10, 0]])['macro_f1'], 0)

    def test_prediction_matches_only_one_target(self):
        _, matches = match_detections([[0, 0, 10, 10, 0.8, 0]],
                                      [[0, 0, 10, 10, 0], [0, 0, 9, 9, 0]], 0.25)
        self.assertEqual(matches, [(0, 0)])

    def test_confidence_filter_changes_false_positives(self):
        p = [[0, 0, 10, 10, 0.9, 0], [20, 20, 30, 30, 0.2, 0]]
        gt = [[0, 0, 10, 10, 0]]
        self.assertAlmostEqual(self.evaluate(p, gt, 0.05)['macro_f1'], 2 / 3)
        self.assertEqual(self.evaluate(p, gt, 0.25)['macro_f1'], 1)

    def test_matching_prefers_higher_iou_not_input_order(self):
        _, matches = match_detections([[0, 0, 9, 9, 0.9, 0], [0, 0, 10, 10, 0.8, 0]],
                                      [[0, 0, 10, 10, 0]], 0.25)
        self.assertEqual(matches, [(1, 0)])

    def test_iou_and_confidence_thresholds_are_inclusive(self):
        r = self.evaluate([[0, 0, 5, 10, 0.25, 0]], [[0, 0, 10, 10, 0]], 0.25)
        self.assertEqual(r['per_class'][0]['tp'], 1)

    def test_macro_f1_is_mean_of_class_f1_not_f1_of_macro_pr(self):
        p = [[0, 0, 10, 10, 0.9, 0], [20, 20, 30, 30, 0.9, 0],
             [40, 40, 50, 50, 0.9, 1]]
        gt = [[0, 0, 10, 10, 0], [40, 40, 50, 50, 1], [60, 60, 70, 70, 1]]
        r = self.evaluate(p, gt)
        self.assertAlmostEqual(r['macro_f1'], 2 / 3)
        harmonic = 2 * r['macro_precision'] * r['macro_recall'] / (r['macro_precision'] + r['macro_recall'])
        self.assertNotAlmostEqual(r['macro_f1'], harmonic)


if __name__ == '__main__':
    unittest.main()
