# PPE Safety Equipment Detection — P14

## Interactive demo

Run `python demo/server.py` and open http://127.0.0.1:8000 to upload images,
compare YOLO11n/YOLO11s detections, adjust confidence, export annotated PNGs,
and inspect saved test results. Install `demo/requirements.txt` for inference;
the saved-results view works without them. See [demo setup and presentation](demo/README.md).

Team plan: [planning/Intro_to_CV_project_plan.pdf](planning/Intro_to_CV_project_plan.pdf).
First working model report: [planning/Intro_to_CV_project_first_working_model.pdf](planning/Intro_to_CV_project_first_working_model.pdf).
Second approach report: [planning/intro_to_cv_3.pdf](planning/intro_to_cv_3.pdf).
Andrei's tasks (Member 1): [planning/member_1.md](planning/member_1.md).
Evgeniy's tasks (Member 2): [planning/member_2.md](planning/member_2.md).
Oleg's tasks (Member 3): [planning/member_3.md](planning/member_3.md).
Dataset preparation results: [reports/dataset/README.md](reports/dataset/README.md).

## Reproducing dataset preparation

Requires Python 3.11+; dataset preparation was tested with Python 3.14.7, training with
Python 3.11.1 on Apple M4 (baseline) and Python 3.14.7 on Apple M2 Max (comparison). Run from the repository root:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/download_dataset.py
python scripts/prepare_dataset.py --apply
```

The first script downloads the source archive (~178 MB), verifies its SHA-256 checksum,
and extracts it. The second applies the saved review decisions and creates a YOLO dataset:
`data/prepared/construction-ppe/dataset.yaml`.
Images retain their original resolution; the model's data loader handles RGB conversion
and letterboxing to 640 pixels. Preparation is repeatable: only
`data/prepared/construction-ppe/` is rebuilt.
The source archive and images are excluded from Git; source details and checksums are
saved in `data/source.json`. The preparation script also generates contact sheets in
`reports/dataset/review_images/` and `reports/dataset/occlusion_images/` (excluded from Git).

## Training the YOLO11n baseline (Member 2)

After the dataset is prepared, train the baseline from the repository root. The protocol
(50 epochs, batch 16, SGD lr0=0.01, momentum=0.937, weight decay=0.0005, 3 warmup epochs,
cosine decay, flips plus mild scale/translate/colour only) is fixed in `scripts/train_yolo11n.py`.

```sh
source .venv/bin/activate
python scripts/train_yolo11n.py               # seeds 42 43 44, device auto (cuda > mps > cpu)
python scripts/train_yolo11n.py --seeds 42 --epochs 2 --project reports/training/smoke
python scripts/train_yolo11n.py --batch 8 --project reports/training/batch8
```

Existing nonempty run folders and saved summaries are protected from accidental overwrite.
Use a fresh `--project` directory for a new experiment; `--resume` is for unfinished runs.
A repeated smoke run also needs a fresh directory. Both notebooks keep training opt-in.

Artifacts per seed land in `reports/training/yolo11n_seed{seed}/` (weights/best.pt selected
by validation mAP50:95, results.csv, curves), and mean/std across seeds with versions,
device, and training time in `reports/training/yolo11n_summary.json`. The research notebook
is `notebooks/001__yolo11n_baseline.ipynb`; register its kernel once and execute it with:

```sh
python -m ipykernel install --user --name cv-project --display-name "Python (cv-project)"
jupyter nbconvert --to notebook --execute --inplace notebooks/001__yolo11n_baseline.ipynb
```

The original baseline was trained on Apple M4 (MPS); `best.pt` checkpoints are
committed, `last.pt` is not.

## Training the YOLO11s comparison (Member 3)

The comparison entry point reuses the baseline implementation and changes only the default
pretrained model. It uses the same frozen split and training protocol:

```sh
python scripts/train_yolo11s.py                 # seeds 42, 43, 44; 50 epochs
python scripts/train_yolo11s.py --jobs 3        # optional concurrent seeds; requires enough GPU memory
python scripts/train_yolo11s.py --seeds 42 --epochs 2 --project reports/training/smoke
```

Outputs mirror the baseline: `reports/training/yolo11s_seed{seed}/` and
`reports/training/yolo11s_summary.json`. The corresponding research notebook is
`notebooks/002__yolo11s_comparison.ipynb`. Its training cells are opt-in, so inspecting
saved results does not start another long run.

The Stage 3 PDF reports an earlier T4 experiment whose original YOLO11s weights/logs
were not retained. Its aggregate numbers are preserved separately in
`reports/evaluation/stage3_reported.json`. Newly trained checkpoints retain their actual
results and environment; the historical numbers are not attached to new weights.
Concurrent training records `training_jobs` in the summary; per-seed wall times include
device contention and are not a controlled training-speed comparison. MPS may report
nondeterministic kernels even when `deterministic=True`; seeds do not guarantee identical bits.
On MPS, `scripts/mps_compat.py` runs the original NMS on CPU and restores outputs to the
GPU, avoiding an observed PyTorch MPS indexing failure. The common evaluation and latency
benchmark use this same path for both models, including the transfer overhead.
On this MPS runtime, the complete original detection criterion and target assignment run
on CPU to avoid sparse-indexing faults. Differentiable device copies preserve model
gradients; model forward/backward remain on GPU. Full-size loss and prediction gradients
are tested against the original CPU criterion. Validation target indexing and metrics also
stay on CPU. See `reports/training/reproduction_notes.md` for the failed attempts and clean rerun.
Interrupted runs with optimizer state can be continued with
`python scripts/train_yolo11s.py --resume` (include only unfinished seeds).
Per-run JSON records the resume checkpoint and completed epoch; native CSV timing restarts
at the resumed segment, while `train_seconds` sums checkpointed prior time and the new segment.

## Validation comparison and Progress check (Stage 4)

Once all six checkpoints exist:

```sh
python scripts/evaluate_models.py --benchmark
```

The evaluator recomputes AP under a common environment, sweeps confidence 0.05–0.90,
selects one threshold per model by mean validation **macro-F1** across seeds, and reports
confidence 0.25 as a reference. It uses class-aware NMS IoU 0.70 and one-to-one same-class
matching at IoU >=0.50. The test split is not loaded. The optional batch-one benchmark
measures both models on the same device after warm-up, including preprocessing and NMS.

Measured outputs are saved in `reports/evaluation/`: a JSON summary, comparison and
per-class CSV tables, and `confidence_sweep.png`. Prediction caches are fingerprinted and
ignored by Git.
See [evaluation protocol](reports/evaluation/README.md) and [Member 3 handoff](planning/member_3.md).

The Results checkpoint below completes test evaluation, size/visibility analysis and
three representative failure cases using the validation-selected settings.

## Results checkpoint (Stage 5)

The completed held-out evaluation and failure analysis are in
[the Results checkpoint report](planning/Intro_to_CV_Results_Checkpoint.pdf).
Reports are retained as final PDFs; document source files and compilation tooling are omitted.
The English report follows the previous submissions. It evaluates all six retained
checkpoints on 78 test images without changing the Stage 4 thresholds (0.20/0.30), and
includes per-class and size/visibility results, controlled CPU latency, training-time
provenance, and three actual annotated failures.

```sh
.venv/bin/python scripts/evaluate_test.py --device cpu --benchmark
.venv/bin/python scripts/build_results_figures.py
```

Test mAP50:95 (mean +/- sample SD) is **0.1874 +/- 0.0037** for YOLO11n and
**0.1902 +/- 0.0067** for YOLO11s. Macro-F1 is **0.4547 +/- 0.0238** and
**0.4835 +/- 0.0218**, respectively. CPU median predict-call latency is 50.21/79.80 ms
on the same M2 Max with four threads, distinct from Stage 4 GPU timings.
See [protocol, outputs and reproduction](reports/results_checkpoint/README.md).

## Final technical summary

The [final technical summary](planning/Intro_to_CV_Final_Technical_Summary.pdf)
summarizes the problem, dataset, methodology, experimental setup, final test results,
limitations and team roles. Annotated comparisons of the three failure cases for both models follow their
discussion, before the conclusions.

## Verification and repository layout

```sh
python -m unittest discover -s tests -v
python -m pip check
```

- `scripts/`: dataset preparation, training, evaluation and figure generation.
- `notebooks/`: inspection of saved experiments; training is opt-in.
- `planning/`: submitted PDF reports and member responsibility notes.
- `reports/`: audited dataset metadata, training logs/checkpoints and evaluation evidence.
- `data/`: source attribution/configuration; downloaded and prepared images are ignored.

CSV/JSON results, six `best.pt` checkpoints, learning curves, confusion matrices and report
figures are retained. Regenerable training/validation batch previews, prediction caches,
local run outputs and the virtual environment are excluded. GitHub Actions runs the unit
tests on CPU; MPS-specific checks skip when MPS is unavailable. Evaluation without
`--benchmark` retains compatible saved latency and its original environment in
`latency_context`; use `--benchmark` to measure the current environment.
