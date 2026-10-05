# PPE Safety Equipment Detection — P14

Team plan: [planning/Intro_to_CV_project_report.pdf](planning/Intro_to_CV_project_report.pdf).
Andrei's tasks (Member 1): [planning/member_1.md](planning/member_1.md).
Dataset preparation results: [reports/dataset/README.md](reports/dataset/README.md).

## Reproducing dataset preparation

Requires Python 3.11+; tested with Python 3.14.7. Run from the repository root:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dataset.txt
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

Members 2 and 3 handle training and metric calculation after the dataset handoff.
