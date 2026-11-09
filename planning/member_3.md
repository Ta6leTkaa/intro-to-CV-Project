# Member 3 - Oleg Bondarev

Your responsibilities are to train the YOLO11s comparison, calculate validation metrics,
select confidence thresholds, and measure inference speed under the same conditions.
Metric definitions are documented for review with Member 2.

## Training environment - completed

- [x] Shared pinned `requirements.txt`, frozen dataset split, and baseline training code.
- [x] `scripts/train_yolo11s.py` reuses the baseline trainer with `yolo11s.pt` as its default.
- [x] Three fresh runs completed on Apple M2 Max (MPS), Python 3.14.7,
  PyTorch 2.14.1 and Ultralytics 8.4.175.
- [x] MPS compatibility: original detection loss and target assignment on CPU,
  model forward/backward on GPU, original NMS and validation metrics on CPU.
  Loss values and gradients were tested against the original CPU implementation.

## YOLO11s comparison - completed

- [x] Same 50-epoch protocol as Member 2: batch 16, imgsz 640, SGD lr0=0.01,
  momentum=0.937, weight decay=0.0005, three warm-up epochs, cosine decay,
  deterministic flag, and seeds 42, 43, 44.
- [x] Same mild flip, scale, translation and HSV augmentation; mosaic, mixup and
  copy-paste disabled. All 11 classes and the frozen 1265/73/78 split retained.
- [x] Best checkpoint selected by validation mAP50-95 for each seed.
- [x] Saved weights, 50-epoch CSV logs, curves, per-run summaries, and aggregate JSON.

Training-time validation mAP50-95: **0.2030 +/- 0.0094**; mAP50: **0.4951 +/- 0.0240**
(mean +/- sample SD). These are the new reproduction results. The earlier T4 results
remain separately attributed to the Stage 3 PDF because its original weights/logs were lost.
All three canonical runs started from pretrained weights at epoch 1 without resuming.
Failed initial MPS attempts are excluded; see `reports/training/reproduction_notes.md`.

## Common validation experiments - completed

- [x] Re-evaluated all six checkpoints in one environment with square 640 letterboxing,
  class-aware NMS IoU 0.70 and confidence 0.001 for AP.
- [x] Swept confidence 0.05–0.90, step 0.05, using actual class macro-F1 and
  one-to-one same-class matching at IoU >=0.50.
- [x] Selected one threshold per model across all seeds: **YOLO11n 0.20; YOLO11s 0.30**.
- [x] Common mAP50-95: **0.1938 +/- 0.0111** (n), **0.2043 +/- 0.0080** (s).
- [x] Selected-threshold macro-F1: **0.4962 +/- 0.0253** (n), **0.5182 +/- 0.0155** (s).
- [x] Controlled batch-one median latency: **7.58 ms** (n), **10.03 ms** (s), same
  M2 Max, FP32, confidence 0.25, 20 warm-up and 100 measured calls. Includes preprocessing,
  inference, CPU NMS, transfers and predict overhead; excludes disk IO and weight loading.
- [x] Test split left untouched. Remaining experiments are listed in the progress report.

## Team handoff

```sh
source .venv/bin/activate
python scripts/train_yolo11s.py
python scripts/evaluate_models.py --benchmark
jupyter nbconvert --to notebook --execute --inplace notebooks/002__yolo11s_comparison.ipynb
```

- Training: `reports/training/yolo11s_seed{42,43,44}/`, including `weights/best.pt`;
  combined metrics and environment: `reports/training/yolo11s_summary.json`.
- Notebook: `notebooks/002__yolo11s_comparison.ipynb`; training and evaluation cells
  are opt-in, while saved tables and plots can be inspected without retraining.
- Comparison, per-class AP/threshold metrics, confidence sweep and latency:
  `reports/evaluation/validation_summary.json` and adjacent CSV/PNG files.
- Subsequent Results checkpoint: held-out test evaluation, size/visibility analysis and
  annotated failures are complete in `reports/results_checkpoint/`; submitted reports
  and the final technical summary are retained as PDFs in `planning/`.
