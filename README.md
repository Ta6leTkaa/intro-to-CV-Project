# PPE Safety Equipment Detection — P14

Team plan: [planning/Intro_to_CV_project_report.pdf](planning/Intro_to_CV_project_report.pdf).
Andrei's tasks (Member 1): [planning/member_1.md](planning/member_1.md).
Evgeniy's tasks (Member 2): [planning/member_2.md](planning/member_2.md).
Dataset preparation results: [reports/dataset/README.md](reports/dataset/README.md).

## Reproducing dataset preparation

Requires Python 3.11+; dataset preparation was tested with Python 3.14.7, training with
Python 3.11.1 on Apple M4. Run from the repository root:

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
python scripts/train_yolo11n.py --epochs 2    # smoke run to validate the setup first
python scripts/train_yolo11n.py --batch 8     # fallback if memory is insufficient
```

Artifacts per seed land in `reports/training/yolo11n_seed{seed}/` (weights/best.pt selected
by validation mAP50:95, results.csv, curves), and mean/std across seeds with versions,
device, and training time in `reports/training/yolo11n_summary.json`. The research notebook
is `notebooks/001__yolo11n_baseline.ipynb`; register its kernel once and execute it with:

```sh
python -m ipykernel install --user --name cv-project --display-name "Python (cv-project)"
jupyter nbconvert --to notebook --execute --inplace notebooks/001__yolo11n_baseline.ipynb
```

Training was performed on Apple M4 (MPS, no CUDA available); `best.pt` checkpoints are
committed, `last.pt` is not. Members 3 and 4 handle YOLO11s, thresholds, metrics, timing,
and failure analysis on the same saved split.
