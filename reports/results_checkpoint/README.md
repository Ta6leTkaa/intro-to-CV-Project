# Results checkpoint (Stage 5)

Completed held-out evaluation of the six retained YOLO11n/YOLO11s checkpoints on the
frozen 78-image test split (924 boxes). Thresholds were selected in Stage 4 on validation
only: YOLO11n 0.20, YOLO11s 0.30. No test sweep or best-test-seed selection is performed.

## Reproduce

Prepare the dataset as described in the root README, retain the six trained best.pt files,
and activate the pinned project environment. From the repository root:

```sh
.venv/bin/python scripts/evaluate_test.py --device cpu --benchmark
.venv/bin/python scripts/build_results_report.py
cd planning/latex
tectonic --outdir .. results_checkpoint.tex
# Or: pdflatex results_checkpoint.tex (twice)
```

The submitted PDF is `planning/Intro_to_CV_Results_Checkpoint.pdf`; the editable entry point
is `planning/latex/results_checkpoint.tex`. Generated tables share its directory. Figure
paths are relative to that directory. `planning/results_checkpoint_latex.zip` contains
self-contained LaTeX sources and figures (compile its root results_checkpoint.tex).
The report is in English to match the four previous submissions.

The evaluator checks every checkpoint against Stage 4 SHA-256 and verifies all 50 epoch
rows. It writes `frozen_protocol.json` before loading test images. A later invocation
refuses changes to this frozen protocol. Prediction caches are ignored by Git and
fingerprinted by checkpoint, test bytes, versions, thread count, device and settings;
use `--refresh` to recompute. The optional benchmark is required to regenerate runtime
results; omitting it only evaluates detection metrics. Re-running can change timing.

## Outputs and definitions

- `test_summary.json`: environment, dataset/checkpoint/annotation hashes, aggregate
  and per-seed metrics, complexity and controlled latency.
- `per_seed.csv`, `model_comparison.csv`: final AP50/AP50:95, macro precision/recall/F1,
  plus macro-F1 at reference confidence 0.25. SD uses n-1 across seeds.
- `per_class.csv`: AP, threshold metrics and TP/FP/FN for every class and seed.
- `test_objects.csv`: original source path/label line joined to each prepared box;
  source normalized coordinates and class are verified during the join.
- `subgroup_recall.csv`: all-class size recall and PPE-only visibility/truncation recall,
  both aggregate and per class, with counts and TP/FN per seed. Matching is performed on
  full images before subsetting. These are object-weighted recalls, not macro recall.
- `visibility_size_distribution.csv`: group-size counts. Size is measured after square
  640 letterboxing, with area thresholds 32^2 and 96^2.
- `plot_estimates.csv`: means, sample SD and support underlying the two comparison plots.
- `complexity.csv`: unfused parameter counts and seed-42 checkpoint bytes (decimal MB).
- `figures/class_ap50_comparison.png`, `figures/size_recall_comparison.png`: grouped
  comparisons with sample SD across the three seeds; all values come from the saved CSVs.
- `failure_examples.json`, remaining `figures/`: three visually inspected, traceable failure cases,
  with both models' seed-42 evidence. Yellow is selected GT; cyan is the highest-IoU
  same-class retained prediction. For the two misses, the cyan prediction belongs to
  a neighbouring object and cannot be reused. The localization boxes are unmatched FP.

AP uses standard Ultralytics multi-label validation at confidence 0.001, batch 16.
Fixed-threshold metrics use standard single-label predict outputs, batch one.
Both use FP32, 640 square letterboxing, class-aware NMS IoU 0.70, max_det=300.
Threshold matching is greedy descending IoU, same class, one-to-one at IoU >=0.50.
All 11 classes have test support. AP is independent of the frozen operating threshold.

## Runtime and limitations

New timing: CPU, Apple M2 Max as identified by Ultralytics, four PyTorch threads,
20 warm-up + 100 timed calls, confidence 0.25 and seed 42, same in-memory test image order.
The general hardware helper returned `arm` in the restricted environment; the Ultralytics
CPU label supplies the more specific model name used in the report. Batch-one predict-call
latency includes resize/pad, forward, NMS and overhead; excludes disk IO/weight loading/plots.
The old Stage 4 MPS timings and unlike-device training wall times are separately attributed.
Training was not rerun. Stage 3 lost T4 weights are not part of these results.

Visibility labels are a single assistant review, not independent agreement of two people.
Primary clear/occluded comparison excludes 190 uncertain PPE objects; their diagnostic
recall is exported separately. Visibility and truncation are separate partitions.
Group counts/class/size distributions differ, so this is observational subgroup analysis.
Three training seeds on one small test split do not give uncertainty over new sites.
No claim of statistical significance or deployment safety is made.

Verification: `python -m unittest discover -s tests` includes one-to-one matching,
macro-F1 definitions and subgroup accounting. MPS-only checks skip when MPS is unavailable.
The compiled report was rendered and visually reviewed page by page.
