# Validation comparison and Stage 4 progress check

The generated results in this directory compare saved YOLO11n and YOLO11s checkpoints
on the frozen 73-image validation split. The test split is not loaded by the evaluator.

```sh
python scripts/evaluate_models.py --benchmark
```

Run from the repository root after dataset preparation and all three training runs per model.
Missing checkpoints are reported before inference starts. To recompute prediction caches:

```sh
python scripts/evaluate_models.py --refresh --benchmark
```

The evaluator also requires completed training summaries and exactly 50 logged epochs
per checkpoint. It refuses checkpoints still being written or changed during inference.
Training-time validation uses the library's rectangular geometry; this comparison
explicitly fixes square letterboxing for both models, so AP can differ from training summaries.

Cached predictions and intermediate Ultralytics outputs are excluded from Git. Cache identity
includes checkpoint SHA-256, validation image and label bytes, versions, device and settings.
The result JSON records these identifiers so the figures can be traced to real checkpoints.

## Metric definitions

- AP: Ultralytics validation, confidence 0.001, NMS IoU 0.70, fixed square letterbox 640,
  batch 16, FP32, class-aware NMS, max_det=300.
- Threshold experiment: predictions use the same image size and NMS settings, batch one;
  confidence is swept from 0.05 to 0.90 in increments of 0.05.
- Matching: same-class IoU >=0.50, greedily assigning highest-IoU eligible pairs without
  reusing either a prediction or a ground-truth object. Below-threshold predictions are removed first.
- Class F1: `2*TP/(2*TP+FP+FN)`; macro-F1 averages class F1 values, not aggregate P/R.
  Precision is defined as zero when TP+FP=0; a ground-truth-present class with no
  detections receives recall=0 and F1=0.
  Macro precision and recall use the same ground-truth-present class set. Absent-class
  false positives are still counted and exported.
- Selection: one threshold per model, maximizing mean macro-F1 across its three seeds;
  exact ties favour the lower threshold. Confidence 0.25 is a reference operating point.
- Variability: sample standard deviation across seeds, not a confidence interval.

AP is ranked-prediction performance; fixed-threshold precision/recall/F1 are separate
operating-point metrics. The selected threshold does not change the reported AP.
AP retains Ultralytics validation's multi-label proposal expansion. Threshold metrics
use standard `predict` outputs (best class per proposal), matching the usual inference
interface. Both models use the same policy for each metric; these flags are recorded in JSON.

## Latency

The optional benchmark uses both models on one device, seed-42 checkpoints, batch one,
FP32, confidence 0.25, 20 warm-up calls and 100 timed calls. It cycles through the same
in-memory validation images in a fixed order and synchronizes the accelerator around each
call. Median, 25th and 75th percentiles are reported. Measurements include resize/padding,
forward inference, NMS, result construction and predict-call overhead. File reading,
checkpoint loading and visualization are excluded. The benchmark is not a comparison
between the old M4 and T4 log timings.
On MPS the original NMS executes on CPU through `scripts/mps_compat.py`, with outputs
restored to MPS. This avoids the observed PyTorch multi-label NMS indexing fault.
Both models use the same workaround, and timings include CPU/GPU transfers.

## Historical Stage 3 report

`stage3_reported.json` contains only values transcribed from the submitted PDF, with
source provenance. The original YOLO11s checkpoint artifacts were lost. Those reported
values are kept separate from the new reproduction runs and the common validation
re-evaluation. They must not be used to label new checkpoints or fabricate per-seed logs.

Output tables: `model_comparison.csv`, `per_class_ap.csv`, `per_class_thresholds.csv`,
`confidence_sweep.csv` and `confidence_summary.csv`. The JSON also records latency
median/IQR, checkpoint hashes and the complete evaluation settings.
