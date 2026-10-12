"""Fine-tune the YOLO11n baseline and validate it under the shared team protocol."""
import argparse
import json
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/prepared/construction-ppe/dataset.yaml'

# Locked by the team report, section 3. The optimizer must be named explicitly
# because the Ultralytics default 'auto' ignores lr0 and momentum.
PROTOCOL = dict(
    imgsz=640,
    optimizer='SGD',
    lr0=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3.0,
    cos_lr=True,
    val=True,
    plots=True,
    deterministic=True,
    # The report limits training augmentation to horizontal flips plus mild
    # geometric and colour jitter, so mosaic, mixup and copy-paste stay off.
    fliplr=0.5,
    degrees=0.0,
    flipud=0.0,
    scale=0.2,
    translate=0.1,
    hsv_h=0.015,
    hsv_s=0.4,
    hsv_v=0.3,
    mosaic=0.0,
    mixup=0.0,
    copy_paste=0.0,
)

# Evaluation settings from the report: class-aware NMS IoU 0.70 and the
# low confidence cutoff of 0.001 for AP; test data is never used here.
VAL = dict(imgsz=640, conf=0.001, iou=0.7, agnostic_nms=False)


def pick_device(requested):
    import torch
    if requested != 'auto':
        return requested
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def collect_metrics(metrics):
    """Flatten an Ultralytics DetMetrics object into JSON-serializable values."""
    names = {int(i): str(n) for i, n in metrics.names.items()}
    per_class_ap50 = {names[i]: None for i in names}
    ap50, ap_index = metrics.box.ap50, metrics.box.ap_class_index
    if ap50 is not None and ap_index is not None:
        for value, cls in zip(ap50, ap_index):
            per_class_ap50[names[int(cls)]] = round(float(value), 4)
    return {
        'mAP50-95': round(float(metrics.box.map), 4),
        'mAP50': round(float(metrics.box.map50), 4),
        'precision': round(float(metrics.box.mp), 4),
        'recall': round(float(metrics.box.mr), 4),
        'per_class_ap50': per_class_ap50,
        'per_class': metrics.summary(),
        'speed_ms_per_image': {k: round(float(v), 2) for k, v in metrics.speed.items()},
    }


def run_seed(seed, epochs, batch, device, project, model_path):
    from ultralytics import YOLO
    run_dir = project / f'{Path(model_path).stem}_seed{seed}'
    start = time.monotonic()
    YOLO(model_path).train(
        data=str(DATA), epochs=epochs, batch=batch, seed=seed, device=device,
        project=str(project), name=run_dir.name, exist_ok=True, **PROTOCOL)
    train_seconds = time.monotonic() - start
    # Since Ultralytics 8.4 the saved best.pt is selected by validation mAP50-95.
    best = run_dir / 'weights/best.pt'
    metrics = YOLO(str(best)).val(data=str(DATA), device=device,
                                  save_dir=str(run_dir / 'val'), plots=False, **VAL)
    return {'seed': seed, 'train_seconds': round(train_seconds, 1),
            'epochs': epochs, 'batch': batch, 'device': device,
            'run_dir': run_dir.relative_to(ROOT).as_posix(),
            'metrics': collect_metrics(metrics)}


def summarize(runs):
    metrics = ['mAP50-95', 'mAP50', 'precision', 'recall']
    aggregate = {}
    for name in metrics:
        values = [r['metrics'][name] for r in runs]
        aggregate[name] = {'mean': round(statistics.fmean(values), 4),
                           'std': round(statistics.stdev(values), 4) if len(values) > 1 else None}
    return aggregate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', default='42,43,44', help='comma separated run seeds')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch', type=int, default=16, help='drop to 8 if memory is insufficient')
    parser.add_argument('--device', default='auto', choices=['auto', 'cuda', 'mps', 'cpu'])
    parser.add_argument('--project', default='reports/training')
    # Member 3 reruns the identical protocol for the comparison model.
    parser.add_argument('--model', default='yolo11n.pt', help='pretrained checkpoint, e.g. yolo11s.pt')
    args = parser.parse_args()
    if not DATA.exists():
        raise SystemExit('Prepared dataset missing. Run scripts/download_dataset.py and '
                         'scripts/prepare_dataset.py --apply first.')
    import torch
    import ultralytics
    project = ROOT / args.project
    project.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    stem = Path(args.model).stem
    runs = []
    for seed in [int(s) for s in args.seeds.split(',') if s.strip()]:
        print(f'=== Training {stem} seed {seed} on {device} ===', flush=True)
        run = run_seed(seed, args.epochs, args.batch, device, project, args.model)
        runs.append(run)
        print(json.dumps(run['metrics'], indent=2, default=float), flush=True)
    summary = {
        'generated_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'data': DATA.relative_to(ROOT).as_posix(),
        'device': device,
        'versions': {'python': platform.python_version(), 'torch': torch.__version__,
                     'ultralytics': ultralytics.__version__, 'platform': platform.platform()},
        'protocol': {'epochs': args.epochs, 'batch': args.batch, **PROTOCOL},
        'runs': runs,
        'aggregate': summarize(runs),
    }
    out = project / f'{stem}_summary.json'
    out.write_text(json.dumps(summary, indent=2, default=float) + '\n')
    print(f'Summary written to {out}')
    print(f"mAP50-95 per seed: {[r['metrics']['mAP50-95'] for r in runs]}")
    print(f"aggregate: {summary['aggregate']['mAP50-95']}")


if __name__ == '__main__':
    main()
