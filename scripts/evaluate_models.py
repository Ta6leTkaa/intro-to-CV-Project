"""Compare saved YOLO checkpoints on validation data; select confidence without test data.

Example: python scripts/evaluate_models.py --models yolo11n yolo11s --benchmark
Prediction caches are fingerprinted by checkpoint, image/label content and environment.
"""
import argparse
import csv
import hashlib
import json
import os
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/cv-project-matplotlib')
os.environ.setdefault('XDG_CACHE_HOME', '/tmp/cv-project-cache')
import numpy as np
from PIL import Image

from benchmark_history import benchmark_context, retain_benchmark
from detection_metrics import evaluate_threshold
from mps_compat import configure_mps_nms, configure_mps_validation
from train_yolo11n import DATA, ROOT, hardware_name, pick_device

OUT = ROOT / 'reports/evaluation'
THRESHOLDS = [i / 100 for i in range(5, 91, 5)]
SETTINGS = dict(imgsz=640, conf=0.001, iou=0.7, agnostic_nms=False,
                max_det=300, rect=False, quantize=32)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def mean_std(values):
    return dict(mean=statistics.fmean(values),
                std=statistics.stdev(values) if len(values) > 1 else None)


def dataset(split='val'):
    if split not in {'val', 'test'}:
        raise ValueError('Only frozen validation/test splits are supported')
    import yaml
    config = yaml.safe_load(DATA.read_text())
    names = config['names']
    names = [names[i] for i in range(len(names))] if isinstance(names, dict) else names
    base = Path(config['path'])
    images = sorted((base / config[split]).glob('*'))
    records = []
    hasher = hashlib.sha256()
    for p in images:
        if not p.is_file() or p.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}:
            continue
        label = base / f'labels/{split}' / (p.stem + '.txt')
        with Image.open(p) as im:
            w, h = im.size
        targets = []
        for line in label.read_text().splitlines():
            if not line.strip():
                continue
            cls, cx, cy, bw, bh = map(float, line.split())
            targets.append([(cx - bw / 2) * w, (cy - bh / 2) * h,
                            (cx + bw / 2) * w, (cy + bh / 2) * h, int(cls)])
        hasher.update(p.name.encode())
        hasher.update(p.read_bytes())
        hasher.update(label.read_bytes())
        records.append(dict(image=p.relative_to(ROOT).as_posix(), targets=targets))
    expected = {'val': 73, 'test': 78}[split]
    if len(records) != expected:
        raise ValueError(f'Expected the frozen {expected}-image {split} split, got {len(records)}')
    return names, records, hasher.hexdigest()


def predict_checkpoint(checkpoint, model_name, seed, device, records, names, data_hash, versions, refresh,
                       split='val', output_dir=None):
    from ultralytics import YOLO
    output_dir = OUT if output_dir is None else Path(output_dir)
    cache_path = output_dir / 'cache' / f'{model_name}_seed{seed}_{split}.json'
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    fingerprint = dict(checkpoint_sha256=digest(checkpoint), dataset_sha256=data_hash,
                       versions=versions, device=device, settings=SETTINGS,
                       class_names=names, ap_batch=16, prediction_batch=1,
                       ap_multilabel=True, prediction_multilabel=False,
                       postprocess_backend=configure_mps_nms(device),
                       validation_backend=configure_mps_validation(device))
    if not refresh and cache_path.exists():
        cached = json.loads(cache_path.read_text())
        if cached['fingerprint'] == fingerprint:
            print(f'Using verified cache: {model_name} seed {seed}', flush=True)
            return cached
    model = YOLO(str(checkpoint))
    if [model.names[i] for i in range(len(model.names))] != names:
        raise ValueError(f'Checkpoint class map differs from the prepared dataset: {checkpoint}')
    images = []
    predictions = model.predict(source=[str(ROOT / r['image']) for r in records],
                                device=device, batch=1, stream=True, verbose=False,
                                save=False, **SETTINGS)
    for source, result in zip(records, predictions, strict=True):
        if Path(result.path).resolve() != (ROOT / source['image']).resolve():
            raise ValueError('Prediction order differs from source/target order; refusing misaligned metrics.')
        boxes = result.boxes
        pred = np.column_stack((boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy(),
                                boxes.cls.cpu().numpy())).tolist()
        images.append(dict(**source, predictions=pred))
    # Re-evaluate AP in the same environment and at the same image geometry for both models.
    metrics = model.val(data=str(DATA), split=split, device=device, batch=16,
                        plots=False, verbose=False, project=str(output_dir / 'cache'),
                        name=f'{model_name}_seed{seed}_ap', exist_ok=True, **SETTINGS)
    per_class = metrics.summary()
    if digest(checkpoint) != fingerprint['checkpoint_sha256']:
        raise RuntimeError('Checkpoint changed during evaluation; finish training before comparing models.')
    cached = dict(fingerprint=fingerprint, images=images,
                  ap=dict(map50_95=float(metrics.box.map), map50=float(metrics.box.map50),
                          per_class=per_class))
    cache_path.write_text(json.dumps(cached, indent=2, default=float) + '\n')
    return cached


def benchmark(checkpoint, records, device, confidence, warmup, repeats):
    """Median batch-one predict call: RGB input conversion, resize, forward pass and NMS.

Input arrays are resident in memory; disk IO, weight loading, and plotting are excluded.
Each model receives the same ordered validation images and identical settings.
"""
    import torch
    from ultralytics import YOLO
    model = YOLO(str(checkpoint))
    samples = []
    for record in records:
        with Image.open(ROOT / record['image']) as im:
            samples.append(np.asarray(im.convert('RGB'))[:, :, ::-1].copy())
    settings = dict(SETTINGS, conf=confidence)
    def synchronize():
        if device == 'mps':
            torch.mps.synchronize()
        elif device.startswith('cuda') or device.isdigit():
            torch.cuda.synchronize(device=int(device) if device.isdigit() else device)
    timings = []
    for i in range(warmup + repeats):
        source = samples[i % len(samples)]
        synchronize()
        start = time.perf_counter()
        model.predict(source=source, device=device, batch=1, verbose=False, save=False, **settings)
        synchronize()
        if i >= warmup:
            timings.append((time.perf_counter() - start) * 1000)
    return dict(median_ms=statistics.median(timings), p25_ms=float(np.percentile(timings, 25)),
                p75_ms=float(np.percentile(timings, 75)), warmup=warmup, repeats=repeats,
                confidence=confidence, batch=1, precision='FP32', device=device,
                includes='predict-call overhead, resize/pad, model forward, NMS and result construction',
                excludes='disk IO, checkpoint loading, plotting')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', default=['yolo11n', 'yolo11s'])
    parser.add_argument('--seeds', default='42,43,44')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--refresh', action='store_true')
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--repeats', type=int, default=100)
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(',')]
    if not seeds or len(set(seeds)) != len(seeds) or len(set(args.models)) != len(args.models):
        parser.error('Use unique models and seeds.')
    if args.warmup < 0 or args.repeats < 1:
        parser.error('Warm-up must be nonnegative and repeats must be positive.')
    # Fail before inference if any requested trained checkpoint is missing.
    checkpoints = {(name, seed): ROOT / f'reports/training/{name}_seed{seed}/weights/best.pt'
                   for name in args.models for seed in seeds}
    missing = [p.relative_to(ROOT).as_posix() for p in checkpoints.values() if not p.exists()]
    if missing:
        raise SystemExit('Trained checkpoints missing:\n' + '\n'.join(missing))
    # A best.pt appears early during training; its existence alone is not completion.
    for name in args.models:
        training_path = ROOT / f'reports/training/{name}_summary.json'
        if not training_path.exists():
            raise SystemExit(f'Completed training summary missing: {training_path.relative_to(ROOT)}')
        training = json.loads(training_path.read_text())
        completed = {r['seed']: r for r in training['runs']}
        for seed in seeds:
            if seed not in completed or completed[seed]['epochs'] != 50:
                raise SystemExit(f'{name} seed {seed} must complete the shared 50-epoch protocol first.')
            with (checkpoints[name, seed].parents[1] / 'results.csv').open() as f:
                epochs = [int(row['epoch']) for row in csv.DictReader(f)]
            if epochs != list(range(1, 51)):
                raise SystemExit(f'{name} seed {seed} has incomplete or duplicated epoch logs.')
    import torch
    import ultralytics
    device = pick_device(args.device)
    postprocess_backend = configure_mps_nms(device)
    validation_backend = configure_mps_validation(device)
    versions = dict(python=platform.python_version(), torch=torch.__version__,
                    ultralytics=ultralytics.__version__, platform=platform.platform(),
                    hardware=hardware_name(device))
    names, records, data_hash = dataset()
    OUT.mkdir(parents=True, exist_ok=True)
    runs, sweeps, per_class_rows, ap_rows = [], [], [], []
    for name in args.models:
        for seed in seeds:
            print(f'Evaluating {name} seed {seed} on validation only', flush=True)
            cached = predict_checkpoint(checkpoints[name, seed], name, seed, device, records, names,
                                        data_hash, versions, args.refresh)
            for c in cached['ap']['per_class']:
                ap_rows.append(dict(model=name, seed=seed, class_name=c['Class'],
                                    instances=c['Instances'], ap50=c['mAP50'], ap50_95=c['mAP50-95']))
            values = []
            for threshold in THRESHOLDS:
                result = evaluate_threshold(cached['images'], threshold, names)
                values.append(result)
                sweeps.append(dict(model=name, seed=seed, confidence=threshold,
                                   macro_precision=result['macro_precision'],
                                   macro_recall=result['macro_recall'], macro_f1=result['macro_f1']))
                per_class_rows += [dict(model=name, seed=seed, confidence=threshold, **r)
                                   for r in result['per_class']]
            runs.append(dict(model=name, seed=seed, checkpoint=checkpoints[name, seed].relative_to(ROOT).as_posix(),
                             checkpoint_sha256=cached['fingerprint']['checkpoint_sha256'],
                             ap=cached['ap'], sweep=values))
    comparison, selected, threshold_summary = [], {}, []
    for name in args.models:
        model_runs = [r for r in runs if r['model'] == name]
        means = {t: statistics.fmean(next(v['macro_f1'] for v in r['sweep'] if v['confidence'] == t)
                                    for r in model_runs) for t in THRESHOLDS}
        # One shared model threshold, chosen by mean validation macro-F1 across seeds.
        # Exact ties prefer the smaller threshold (higher-recall operating point).
        chosen = max(THRESHOLDS, key=lambda t: (means[t], -t))
        selected[name] = chosen
        for t in THRESHOLDS:
            values = [next(v['macro_f1'] for v in r['sweep'] if v['confidence'] == t) for r in model_runs]
            threshold_summary.append(dict(model=name, confidence=t, **mean_std(values)))
        row = dict(model=name, seeds=len(model_runs), confidence=chosen)
        for key in ['map50_95', 'map50']:
            aggregate = mean_std([r['ap'][key] for r in model_runs])
            row.update({key + '_' + k: v for k, v in aggregate.items()})
        for key in ['macro_precision', 'macro_recall', 'macro_f1']:
            for suffix, t in [('', chosen), ('_ref025', 0.25)]:
                aggregate = mean_std([next(v[key] for v in r['sweep'] if v['confidence'] == t)
                                      for r in model_runs])
                row.update({key + suffix + '_' + k: v for k, v in aggregate.items()})
        comparison.append(row)
    write_csv(OUT / 'confidence_sweep.csv', sweeps)
    write_csv(OUT / 'confidence_summary.csv', threshold_summary)
    write_csv(OUT / 'per_class_thresholds.csv', per_class_rows)
    write_csv(OUT / 'per_class_ap.csv', ap_rows)
    write_csv(OUT / 'model_comparison.csv', comparison)
    benchmark_hashes = {name: digest(checkpoints[name, seeds[0]]) for name in args.models}
    latencies, latency_context = retain_benchmark(OUT / 'validation_summary.json', data_hash,
                                                  benchmark_hashes, 'val')
    if args.benchmark:
        latencies = {}
        # Fixed seed and fixed confidence isolate model latency from seed/threshold selection.
        for name in args.models:
            latencies[name] = benchmark(checkpoints[name, seeds[0]], records, device, 0.25,
                                        args.warmup, args.repeats)
        latency_context = benchmark_context(data_hash, benchmark_hashes, 'val', versions, device,
                                            datetime.now(timezone.utc).isoformat())
    summary = dict(generated_utc=datetime.now(timezone.utc).isoformat(timespec='seconds'),
                   split='val', images=len(records), dataset_sha256=data_hash, versions=versions,
                   device=device, prediction_settings=SETTINGS, matching_iou=0.5,
                   ap_multilabel=True, threshold_prediction_multilabel=False,
                   ap_batch=16, threshold_prediction_batch=1,
                   postprocess_backend=postprocess_backend,
                   validation_backend=validation_backend,
                   matching='same-class greedy descending IoU, one prediction and one target at most once',
                   macro_average='class F1/precision/recall averaged over classes with validation ground truth',
                   selection='one threshold per model maximizing mean validation macro-F1 across seeds; lower threshold wins ties',
                   selected_thresholds=selected, test_used=False, runs=runs,
                   comparison=comparison, latency=latencies, latency_context=latency_context)
    (OUT / 'validation_summary.json').write_text(json.dumps(summary, indent=2, default=float) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    for name in args.models:
        rr = [r for r in threshold_summary if r['model'] == name]
        x = np.array([r['confidence'] for r in rr]);y = np.array([r['mean'] for r in rr])
        std = np.array([r['std'] or 0 for r in rr])
        line, = ax.plot(x, y, marker='o', markersize=3, label=name)
        ax.fill_between(x, y - std, y + std, alpha=0.15, color=line.get_color())
        ax.axvline(selected[name], alpha=0.3, linestyle='--', color=line.get_color())
    ax.set(xlabel='Confidence threshold', ylabel='Validation macro-F1', ylim=(0, 1))
    ax.grid(alpha=0.25);ax.legend();fig.tight_layout()
    fig.savefig(OUT / 'confidence_sweep.png', dpi=180);plt.close(fig)
    print(json.dumps(dict(selected_thresholds=selected, comparison=comparison, latency=latencies), indent=2))


if __name__ == '__main__':
    main()
