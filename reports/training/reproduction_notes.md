# YOLO11s reproduction provenance

The Stage 3 PDF reports Tesla T4 experiments with Ultralytics 8.4.182. Their original
weights and per-seed logs were not retained. Historical aggregate values are preserved
separately in `reports/evaluation/stage3_reported.json`; they are not measurements of
new checkpoints and the submitted PDF remains unchanged.

## Canonical Stage 4 runs

The canonical `yolo11s_seed42`, `yolo11s_seed43`, and `yolo11s_seed44` directories are
**fresh runs from COCO-pretrained weights**, with the original 50-epoch budget, batch 16,
11 classes, frozen 1265/73/78 split, SGD settings and augmentation. They start from epoch
1, rather than inheriting weights from the interrupted attempts described below.

Hardware/software: Apple M2 Max, Python 3.14.7, PyTorch 2.14.1, Ultralytics 8.4.175.
Three independent seed processes share the device (`--jobs 3`). Per-seed wall times
include contention, so they are not a controlled training-speed benchmark.

On this runtime, `scripts/mps_compat.py` uses the **complete original detection criterion
and target assignment on CPU**, with differentiable device copies. Neural-network
forward/backward and model parameters remain on MPS. The objective and hyperparameters
are unchanged. Tests compare original CPU and hybrid loss values and prediction gradients
at a realistic 8400-anchor, 11-class head shape, for ordinary and empty targets.

NMS uses the original Ultralytics algorithm on CPU, and validation target indexing and
metric matching also stay on CPU. The same evaluation/NMS path is applied to both models.
The inference benchmark includes CPU/GPU transfers but no training loss calculation.

## Why the native MPS attempts were excluded

The initial parallel attempt failed in native MPS multi-label NMS (`torch.where`) with
an invalid index of 5139361840434986294 for an axis of size 8379. Optimizer checkpoints
were initially resumed after epochs 10/9/20. NMS on CPU passed direct equivalence tests,
but native MPS `nonzero` then failed in the foreground loss with an invalid requested
buffer size of 1077952576.25 GiB.

Narrow CPU-mask adapters passed small loss/backward tests, but a subsequent larger
training assignment still failed with an invalid index. Those adaptations were replaced
by the complete original CPU criterion and CPU metric bookkeeping. A full training epoch
and realistic-shape loss/gradient/validation tests passed before clean canonical training.

All interrupted attempts are kept locally under `reports/training/interrupted/` and
excluded from Git and from the model comparison. Their native CSV histories include
resume time resets; they must not be mixed with the clean final logs. The general
`--resume` option remains available for future interruptions, with per-run recovery
metadata and immediate failure logs.

PyTorch may report nondeterministic MPS kernels despite `deterministic=True`. Seeds,
versions, data hashes and checkpoint hashes are recorded, but bitwise identity across
hardware and runtime versions is not promised.
