"""MPS compatibility: original CPU detection criterion and CPU NMS.

PyTorch 2.14.1 on M2 Max raised invalid-index errors in native MPS NMS and
foreground assignment/loss indexing. The complete, unchanged detection criterion
runs on CPU with differentiable device copies; model forward/backward stay on MPS.
NMS also runs on CPU for both models' common evaluation and timing.
"""
from functools import wraps


def configure_mps_training_loss(device):
    """Use the original detection criterion on CPU with autograd across device copies.

    Only the criterion/target assignment moves to CPU. Model convolutions and their
    backward remain on MPS. This avoids all observed MPS sparse-indexing failures,
    rather than adapting individual index calls. Loss values and prediction gradients
    are checked against the original CPU criterion on realistic 640-pixel head shapes.
    """
    if device != 'mps':
        return 'native'
    import torch
    from ultralytics.utils.loss import v8DetectionLoss
    if getattr(v8DetectionLoss.__call__, '_cv_cpu_criterion', False):
        return 'cpu_detection_criterion_with_mps_model'
    original = v8DetectionLoss.__call__

    def move(value, target):
        if isinstance(value, torch.Tensor):
            return value.to(target)
        if isinstance(value, dict):
            return {key: move(item, target) for key, item in value.items()}
        if isinstance(value, list):
            return [move(item, target) for item in value]
        if isinstance(value, tuple):
            return tuple(move(item, target) for item in value)
        return value

    @wraps(original)
    def cpu_criterion(self, preds, batch):
        parsed = self.parse_output(preds)
        output_device = parsed['boxes'].device
        if output_device.type != 'mps':
            return original(self, preds, batch)
        if self.device.type != 'cpu':
            for name, value in list(vars(self).items()):
                if isinstance(value, torch.Tensor):
                    setattr(self, name, value.cpu())
                elif isinstance(value, torch.nn.Module):
                    value.cpu()
            self.device = torch.device('cpu')
        cpu_preds = move(parsed, 'cpu')
        cpu_batch = dict(batch)
        for name in ('batch_idx', 'cls', 'bboxes'):
            cpu_batch[name] = batch[name].cpu()
        loss, items = original(self, cpu_preds, cpu_batch)
        return move(loss, output_device), move(items, output_device)

    cpu_criterion._cv_cpu_criterion = True
    v8DetectionLoss.__call__ = cpu_criterion
    return 'cpu_detection_criterion_with_mps_model'


def configure_mps_nms(device):
    if device != 'mps':
        return 'native'
    import torch
    from ultralytics.utils import nms
    if getattr(nms.non_max_suppression, '_cv_cpu_mps', False):
        return 'cpu_nms_with_mps_transfers'
    original = nms.non_max_suppression

    @wraps(original)
    def cpu_nms(prediction, *args, **kwargs):
        raw = prediction[0] if isinstance(prediction, (list, tuple)) else prediction
        if not isinstance(raw, torch.Tensor) or raw.device.type != 'mps':
            return original(prediction, *args, **kwargs)
        original_device = raw.device
        result = original(raw.cpu(), *args, **kwargs)
        def restore(value):
            if isinstance(value, torch.Tensor):
                return value.to(original_device)
            if isinstance(value, tuple):
                return tuple(restore(item) for item in value)
            if isinstance(value, list):
                return [restore(item) for item in value]
            return value
        return restore(result)

    cpu_nms._cv_cpu_mps = True
    nms.non_max_suppression = cpu_nms
    return 'cpu_nms_with_mps_transfers'


def configure_mps_validation(device):
    """Keep target indexing and metric matching on CPU, while images/model use MPS."""
    if device != 'mps':
        return 'native'
    import torch
    from ultralytics.models.yolo.detect.val import DetectionValidator
    if getattr(DetectionValidator._prepare_batch, '_cv_cpu_targets', False):
        return 'mps_forward_cpu_nms_and_metrics'
    original_batch = DetectionValidator._prepare_batch
    original_postprocess = DetectionValidator.postprocess

    @wraps(original_batch)
    def prepare_batch(self, si, batch):
        if self.device.type != 'mps':
            return original_batch(self, si, batch)
        cpu_batch = dict(batch)
        for key in ('batch_idx', 'cls', 'bboxes'):
            cpu_batch[key] = batch[key].cpu()
        original_device = self.device
        self.device = torch.device('cpu')
        try:
            return original_batch(self, si, cpu_batch)
        finally:
            self.device = original_device

    @wraps(original_postprocess)
    def postprocess(self, preds):
        result = original_postprocess(self, preds)
        if self.device.type == 'mps':
            return [{key: value.cpu() if isinstance(value, torch.Tensor) else value
                     for key, value in item.items()} for item in result]
        return result

    prepare_batch._cv_cpu_targets = True
    DetectionValidator._prepare_batch = prepare_batch
    DetectionValidator.postprocess = postprocess
    return 'mps_forward_cpu_nms_and_metrics'
