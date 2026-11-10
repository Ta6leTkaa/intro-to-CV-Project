# PPE web demo

Run from the repository root, using Python 3.11–3.15:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r demo/requirements.txt
python demo/server.py
```

Open http://127.0.0.1:8000. No Node.js, web framework, frontend build or extra Python
dependencies are required. The server binds to loopback only. Stop it with Ctrl+C.
Use `--port 8001` to change the port. CPU is the default; `--device mps` enables the
project's CPU NMS compatibility path, and `--device cuda` selects CUDA when available.

The UI and saved report work without ML dependencies: `python demo/server.py` can
start before installation. Inference stays disabled until dependencies are installed
in the environment used to launch the server. Refresh the page after installation.

The example picker includes Clear PPE, Multiple workers and Distant workers out of
the box. These bundled demo photos require no download.

The upstream archive has changed; historical test preparation requires a copy
matching `data/source.json`. With that original archive available, use:

```sh
python scripts/download_dataset.py
python scripts/prepare_dataset.py --apply
```

Your own JPEG/PNG/WebP images work without the dataset. Uploads are held in memory
and are not saved to disk or sent to external services. Limit: 15 MB / 24 megapixels.
The service is a local demo, not a public hosting server.

## Interactive detections

Hover or focus a box to highlight it and show its class/confidence. Click or press
Enter/Space to pin the selection. Select an object from the expandable Detections
list to highlight the same box. Click Clear, press Escape on an object, click empty
image space, or click the selected object again to clear it. Each model has its own
selection. Save PNG exports all visible boxes without transient highlighting.

## Suggested presentation

1. Upload a clear construction photo or choose a test example.
2. Select “Both” and run inference. Both models see the same frame.
3. Inspect boxes, classes, confidence and detection counts; export a PNG.
4. Compare the Multiple workers and Distant workers examples.
5. Open “Results” for committed aggregate metrics and annotated failures.

The live view uses seed-42 `best.pt` checkpoints, FP32, imgsz=640, square letterboxing,
class-aware NMS IoU=0.70 and max_det=300. Validation thresholds default to 0.20/0.30;
sliders affect the next run. Live latency times the predict call, excludes initial
weight loading and includes first-call setup/warm-up. It is not the report benchmark.
Saved test metrics are read from the committed CSV, averaged across three seeds;
they are never replaced by metrics from an uploaded photo. No training or evaluation
reports are modified by the demo.

## Verify

```sh
python -m unittest discover -s tests -p 'test_demo.py' -v
```

HTTP tests mock inference to run without torch/GPU. To verify actual inference,
install the demo requirements, upload an image, and run both models in the UI.

## PyTorch installation

The historical torch==2.14.1 pin has no compatible wheel in the project's Python 3.15
environment. The demo requirements allow torch 2.15 and torchvision 0.29.2, which
resolve on macOS ARM with Python 3.15. TorchVision 0.29 is ABI-compatible with
PyTorch 2.14 and later (https://github.com/pytorch/vision/releases).

Use `python -m pip install -r demo/requirements.txt` in the active environment.
The original project dependencies remain in the root `requirements.txt`; the site
uses its separate `demo/requirements.txt`. Saved experiment metrics retain their
original environment. Live demo inference does not reproduce historical timings.
