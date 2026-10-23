"""Class-aware, one-to-one detection matching and explicit macro metrics.

Boxes use original-image xyxy coordinates. Predictions have six columns:
x1, y1, x2, y2, confidence, class_id; targets have five: xyxy, class_id.
"""
import numpy as np


def box_iou(left, right):
    left = np.asarray(left, dtype=float).reshape(-1, 4)
    right = np.asarray(right, dtype=float).reshape(-1, 4)
    intersection = np.maximum(
        np.minimum(left[:, None, 2:], right[None, :, 2:])
        - np.maximum(left[:, None, :2], right[None, :, :2]), 0).prod(axis=2)
    area_l = np.maximum(left[:, 2:] - left[:, :2], 0).prod(axis=1)
    area_r = np.maximum(right[:, 2:] - right[:, :2], 0).prod(axis=1)
    union = area_l[:, None] + area_r[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def match_detections(predictions, targets, confidence, iou_threshold=0.5):
    """Greedily match same-class pairs in descending IoU order, without reuse."""
    predictions = np.asarray(predictions, dtype=float).reshape(-1, 6)
    targets = np.asarray(targets, dtype=float).reshape(-1, 5)
    keep = np.flatnonzero(predictions[:, 4] >= confidence)
    pred = predictions[keep]
    ious = box_iou(pred[:, :4], targets[:, :4])
    valid = (ious >= iou_threshold) & (pred[:, None, 5] == targets[None, :, 4])
    candidates = [(float(ious[p, g]), int(p), int(g)) for p, g in zip(*np.nonzero(valid))]
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    matched_p, matched_g, matches = set(), set(), []
    for _, p, g in candidates:
        if p not in matched_p and g not in matched_g:
            matched_p.add(p)
            matched_g.add(g)
            matches.append((int(keep[p]), g))
    return keep, matches


def evaluate_threshold(images, confidence, names):
    counts = np.zeros((len(names), 3), dtype=int)  # TP, FP, FN
    support = np.zeros(len(names), dtype=int)
    for image in images:
        pred = np.asarray(image['predictions'], dtype=float).reshape(-1, 6)
        gt = np.asarray(image['targets'], dtype=float).reshape(-1, 5)
        kept, matches = match_detections(pred, gt, confidence)
        for cls in range(len(names)):
            n_gt = int(np.sum(gt[:, 4] == cls))
            n_pred = int(np.sum(pred[kept, 5] == cls))
            tp = sum(gt[g, 4] == cls for _, g in matches)
            counts[cls] += (tp, n_pred - tp, n_gt - tp)
            support[cls] += n_gt
    per_class = []
    for cls, name in enumerate(names):
        tp, fp, fn = map(int, counts[cls])
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if support[cls] else None
        f1 = 2 * tp / (2 * tp + fp + fn) if support[cls] else None
        per_class.append(dict(class_id=cls, class_name=name, support=int(support[cls]),
                              tp=tp, fp=fp, fn=fn, precision=precision, recall=recall, f1=f1))
    present = [r for r in per_class if r['support']]
    mean = lambda key: float(np.mean([r[key] for r in present])) if present else None
    return dict(confidence=float(confidence), macro_precision=mean('precision'),
                macro_recall=mean('recall'), macro_f1=mean('f1'), per_class=per_class,
                absent_class_false_positives=sum(r['fp'] for r in per_class if not r['support']))
