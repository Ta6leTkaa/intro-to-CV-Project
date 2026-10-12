# Member 2 - Evgeniy Pestrovskiy

Your responsibilities in the team plan are to set up the training environment, train the
YOLO11n baseline under the shared protocol, and document how to run everything. You also
check metric calculations with Member 3.

## Training environment - completed

- [x] Single pinned `requirements.txt` replaces `requirements-dataset.txt`; it covers the
  dataset scripts, the training pipeline, and the research notebooks.
- [x] `.venv` setup and the exact commands live in the root README.
- [x] Device selection order cuda > mps > cpu; the baseline was trained locally on
  Apple M4 (MPS). No CUDA host was available for this part.

## YOLO11n baseline - completed

- [x] Protocol fixed in `scripts/train_yolo11n.py` from report section 3: 50 epochs,
  batch 16, imgsz 640, SGD lr0=0.01, momentum=0.937, weight decay=0.0005, three warmup
  epochs, cosine LR decay, `deterministic=True`, seeds 42, 43, and 44.
- [x] Augmentation is limited to the report list: horizontal flips 0.5, scale 0.2,
  translate 0.1, mild HSV (0.015 / 0.4 / 0.3); mosaic, mixup, and copy-paste are off.
  Validation and test images are only resized and padded by the loader.
- [x] `best.pt` is selected by validation mAP50-95 (the Ultralytics 8.4+ fitness rule).
- [x] Evaluation uses class-aware NMS IoU 0.70 and confidence cutoff 0.001 on the
  validation split only; the test split was never used for tuning.
- [x] Three seed runs trained; per-run metrics, versions, device, and training time in
  `reports/training/yolo11n_summary.json`.

Result: mAP50-95 per seed 42 / 43 / 44: **0.1895 / 0.2023 / 0.1934**, mean **0.1951**,
standard deviation **0.0066**. mAP50: 0.4760 +/- 0.0185, precision: 0.5807 +/- 0.0579,
recall: 0.4661 +/- 0.0163 (validation split, NMS IoU 0.70, conf 0.001). Each run took
about 51 minutes on the M4 (Apple MPS). Per-class AP50 and the full table are in the
summary JSON; the validation curves peak at epochs 37 / 45 / 36 and drift down afterwards,
so the run converges within the fixed 50-epoch budget. The weakest classes are no_boots
(AP50 about 0.06) and goggles (about 0.22), which the 73-image validation split measures
noisily - Members 3 and 4 should read them cautiously.

## Team handoff

- Members 3 and 4: run the identical protocol for the comparison model with
  `python scripts/train_yolo11n.py --model yolo11s.pt`; checkpoints and summaries land
  in `reports/training/yolo11s_seed{N}/` and `reports/training/yolo11s_summary.json`.
- Checkpoints: `reports/training/yolo11n_seed{N}/weights/best.pt` (committed); per-run
  logs, curves, and confusion matrices are next to them; `last.pt` is not committed.
- The research notebook is `notebooks/001__yolo11n_baseline.ipynb` (run from the
  repository root; the three-seed cell is guarded by `FULL_RUN` to keep reruns fast).
- Threshold selection, macro-F1 at a fixed confidence, latency, and failure analysis
  belong to Members 3 and 4 on this same saved split.
