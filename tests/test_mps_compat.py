"""Verify that the MPS workaround preserves CPU NMS outputs and result devices."""
import sys
import unittest
from pathlib import Path

import torch
from ultralytics.utils import nms

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from mps_compat import configure_mps_nms, configure_mps_training_loss, configure_mps_validation


class MPSCompatibilityTests(unittest.TestCase):
    def predictions(self):
        # Three xywh proposals: duplicate class 0 boxes and one separate class 1 box.
        return torch.tensor([[[5., 5., 25.], [5., 5., 25.], [10., 10., 8.],
                              [10., 10., 8.], [.9, .8, .01], [.01, .02, .85]]])

    def test_cpu_inputs_unchanged_and_configuration_idempotent(self):
        expected = nms.non_max_suppression(self.predictions(), .05, .7, nc=2, multi_label=True)
        configure_mps_nms('mps')
        installed = nms.non_max_suppression
        configure_mps_nms('mps')
        self.assertIs(installed, nms.non_max_suppression)
        actual = nms.non_max_suppression(self.predictions(), .05, .7, nc=2, multi_label=True)
        self.assertEqual(len(actual[0]), 2)
        torch.testing.assert_close(actual[0], expected[0])

    @unittest.skipUnless(torch.backends.mps.is_available(), 'MPS device unavailable')
    def test_mps_predictions_match_cpu_and_restore_device(self):
        configure_mps_nms('mps')
        expected = nms.non_max_suppression(self.predictions(), .05, .7, nc=2, multi_label=True, return_idxs=True)
        actual = nms.non_max_suppression((self.predictions().to('mps'), None), .05, .7,
                                         nc=2, multi_label=True, return_idxs=True)
        self.assertEqual(actual[0][0].device.type, 'mps')
        self.assertEqual(actual[1][0].device.type, 'mps')
        torch.testing.assert_close(actual[0][0].cpu(), expected[0][0])
        torch.testing.assert_close(actual[1][0].cpu(), expected[1][0])

    @unittest.skipUnless(torch.backends.mps.is_available(), 'MPS device unavailable')
    def test_full_cpu_criterion_preserves_loss_and_prediction_gradients(self):
        from types import SimpleNamespace
        from ultralytics.utils.loss import v8DetectionLoss

        class StubModel(torch.nn.Module):
            def __init__(self, device):
                super().__init__()
                self.marker = torch.nn.Parameter(torch.zeros(1, device=device))
                self.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)
                self.model = [SimpleNamespace(stride=torch.tensor([8., 16., 32.], device=device),
                                              nc=11, reg_max=16)]

        configure_mps_training_loss('mps')
        for empty in [False, True]:
            torch.manual_seed(42)
            boxes = torch.randn(2, 64, 8400)
            scores = torch.randn(2, 11, 8400)
            feats = [torch.zeros(2, 1, n, n) for n in [80, 40, 20]]
            batch = dict(batch_idx=torch.tensor([0., 0., 1., 1.]),
                         cls=torch.tensor([[0.], [4.], [10.], [2.]]),
                         bboxes=torch.tensor([[.3, .4, .12, .15], [.7, .6, .18, .15],
                                              [.5, .5, .2, .2], [.2, .7, .1, .18]]))
            if empty:
                batch = {key: value[:0] for key, value in batch.items()}
            cpu_boxes = boxes.clone().requires_grad_(True)
            cpu_scores = scores.clone().requires_grad_(True)
            expected, _ = v8DetectionLoss(StubModel('cpu'))(
                dict(boxes=cpu_boxes, scores=cpu_scores, feats=feats), batch)
            expected.sum().backward()
            gpu_boxes = boxes.to('mps').requires_grad_(True)
            gpu_scores = scores.to('mps').requires_grad_(True)
            gpu_batch = {key: value.to('mps') for key, value in batch.items()}
            criterion = v8DetectionLoss(StubModel('mps'))
            actual, items = criterion(dict(boxes=gpu_boxes, scores=gpu_scores,
                                           feats=[x.to('mps') for x in feats]), gpu_batch)
            self.assertEqual(actual.device.type, 'mps')
            self.assertEqual(criterion.device.type, 'cpu')
            torch.testing.assert_close(actual.cpu(), expected, rtol=1e-5, atol=1e-4)
            actual.sum().backward()
            torch.testing.assert_close(gpu_boxes.grad.cpu(), cpu_boxes.grad, rtol=1e-5, atol=1e-5)
            torch.testing.assert_close(gpu_scores.grad.cpu(), cpu_scores.grad, rtol=1e-5, atol=1e-5)

    @unittest.skipUnless(torch.backends.mps.is_available(), 'MPS device unavailable')
    def test_validation_targets_and_postprocessing_remain_on_cpu(self):
        from ultralytics.models.yolo.detect.val import DetectionValidator
        configure_mps_nms('mps')
        configure_mps_validation('mps')
        validator = DetectionValidator(args=dict(conf=.05, iou=.7, max_det=300))
        validator.device = torch.device('mps')
        validator.end2end = False
        batch = dict(batch_idx=torch.tensor([0., 1.], device='mps'),
                     cls=torch.tensor([[0.], [10.]], device='mps'),
                     bboxes=torch.tensor([[.5, .5, .2, .2]] * 2, device='mps'),
                     img=torch.zeros(2, 3, 640, 640, device='mps'),
                     ori_shape=[(640, 640)] * 2, ratio_pad=[None] * 2,
                     im_file=['a.jpg', 'b.jpg'])
        prepared = validator._prepare_batch(1, batch)
        self.assertEqual(prepared['cls'].device.type, 'cpu')
        self.assertEqual(prepared['cls'].tolist(), [10.])
        torch.testing.assert_close(prepared['bboxes'], torch.tensor([[256., 256., 384., 384.]]))
        self.assertEqual(validator.device.type, 'mps')
        outputs = validator.postprocess(self.predictions().to('mps'))
        self.assertEqual(outputs[0]['bboxes'].device.type, 'cpu')
        self.assertEqual(len(outputs[0]['cls']), 2)


if __name__ == '__main__':
    unittest.main()
