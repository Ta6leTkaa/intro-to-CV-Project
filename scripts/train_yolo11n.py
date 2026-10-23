"""Fine-tune the YOLO11n baseline and validate it under the shared team protocol."""
import argparse
import concurrent.futures
import csv
import hashlib
import json
import multiprocessing
import platform
import statistics
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from mps_compat import configure_mps_nms, configure_mps_training_loss, configure_mps_validation

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


def hardware_name(device):
    """Record the actual hardware without assuming that all MPS hosts are identical."""
    import torch
    if device.startswith('cuda') or device.isdigit():
        return torch.cuda.get_device_name(int(device) if device.isdigit() else device)
    if platform.system() == 'Darwin':
        result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'],
                                capture_output=True, text=True, check=False)
        if result.returncode == 0:
            return result.stdout.strip()
    return platform.processor() or platform.machine()


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


def _run_seed(seed, epochs, batch, device, project, model_path, resume=False):
    from ultralytics import YOLO
    postprocess_backend = configure_mps_nms(device)
    training_loss_backend = configure_mps_training_loss(device)
    validation_backend = configure_mps_validation(device)
    run_dir = project / f'{Path(model_path).stem}_seed{seed}'
    failure = run_dir / 'failure.log'
    if failure.exists():
        archive = run_dir / 'interrupted_attempts'
        archive.mkdir(exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        failure.replace(archive / f'{stamp}.log')
    resume_info = None
    previous_elapsed = 0.0
    checkpoint = run_dir / 'weights/last.pt'
    if resume:
        import torch
        if not checkpoint.exists():
            raise FileNotFoundError(f'Resume checkpoint missing: {checkpoint}')
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        completed_epoch = int(state['epoch']) + 1
        if completed_epoch <= 0 or completed_epoch >= epochs or state.get('optimizer') is None:
            raise ValueError('Resume requires an unfinished checkpoint with its optimizer state.')
        if int(state['train_args']['epochs']) != epochs or int(state['train_args']['batch']) != batch:
            raise ValueError('Resume epoch budget and batch must match the saved experiment.')
        with (run_dir / 'results.csv').open() as f:
            prior_rows = list(csv.DictReader(f))
        # Ultralytics restarts CSV time on resume; retain every completed segment.
        offset = 0.0
        previous = 0.0
        for row in prior_rows:
            if int(row['epoch']) > completed_epoch:
                break
            current = float(row['time'])
            if current < previous:
                offset += previous
            previous = current
        previous_elapsed = offset + previous
        resume_info = dict(completed_epoch=completed_epoch,
                           checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                           previous_checkpoint_elapsed_seconds=previous_elapsed,
                           note='Resumed optimizer/scheduler state; discarded work after the last saved epoch is excluded from timing.')
        del state
        history_path = run_dir / 'resume_history.json'
        history = json.loads(history_path.read_text()) if history_path.exists() else []
        history.append(resume_info)
        history_path.write_text(json.dumps(history, indent=2) + '\n')
    start = time.monotonic()
    YOLO(str(checkpoint) if resume else model_path).train(
        data=str(DATA), epochs=epochs, batch=batch, seed=seed, device=device,
        project=str(project), name=run_dir.name, exist_ok=True, resume=resume, **PROTOCOL)
    segment_train_seconds = time.monotonic() - start
    train_seconds = previous_elapsed + segment_train_seconds
    # Since Ultralytics 8.4 the saved best.pt is selected by validation mAP50-95.
    best = run_dir / 'weights/best.pt'
    metrics = YOLO(str(best)).val(data=str(DATA), device=device,
                                  save_dir=str(run_dir / 'val'), plots=False, **VAL)
    result = {'seed': seed, 'train_seconds': round(train_seconds, 1),
              'epochs': epochs, 'batch': batch, 'device': device,
              'postprocess_backend': postprocess_backend,
              'training_loss_backend': training_loss_backend,
              'validation_backend': validation_backend,
              'resume': resume_info,
              'resume_history': history if resume else [],
              'segment_train_seconds': round(segment_train_seconds, 1),
              'run_dir': run_dir.relative_to(ROOT).as_posix(),
              'metrics': collect_metrics(metrics)}
    # Preserve each completed run even if a later seed is interrupted.
    (run_dir / 'run_summary.json').write_text(json.dumps(result, indent=2, default=float) + '\n')
    return result


def run_seed(seed, epochs, batch, device, project, model_path, resume=False):
    try:
        return _run_seed(seed, epochs, batch, device, project, model_path, resume)
    except BaseException:
        run_dir = project / f'{Path(model_path).stem}_seed{seed}'
        run_dir.mkdir(parents=True, exist_ok=True)
        detail = traceback.format_exc()
        (run_dir / 'failure.log').write_text(detail)
        print(f'Training seed {seed} failed:\n{detail}', flush=True)
        raise


def summarize(runs):
    metrics = ['mAP50-95', 'mAP50', 'precision', 'recall']
    aggregate = {}
    for name in metrics:
        values = [r['metrics'][name] for r in runs]
        aggregate[name] = {'mean': round(statistics.fmean(values), 4),
                           'std': round(statistics.stdev(values), 4) if len(values) > 1 else None}
    return aggregate


def main(default_model='yolo11n.pt'):
    parser = argparse.ArgumentParser(description=f'Fine-tune {Path(default_model).stem} under the shared team protocol.')
    parser.add_argument('--seeds', default='42,43,44', help='comma separated run seeds')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch', type=int, default=16, help='drop to 8 if memory is insufficient')
    parser.add_argument('--device', default='auto', choices=['auto', 'cuda', 'mps', 'cpu'])
    parser.add_argument('--project', default='reports/training')
    # Member 3 reruns the identical protocol for the comparison model.
    parser.add_argument('--model', default=default_model, help='pretrained checkpoint, e.g. yolo11s.pt')
    parser.add_argument('--jobs', type=int, default=1,
                        help='independent seed processes sharing the device; default 1')
    parser.add_argument('--resume', action='store_true',
                        help='continue every requested seed from its unfinished last.pt')
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
    seeds = [int(s) for s in args.seeds.split(',') if s.strip()]
    if not seeds or len(set(seeds)) != len(seeds) or args.jobs < 1:
        parser.error('Use unique seeds and a positive number of jobs.')
    runs = []
    jobs = min(args.jobs, len(seeds))
    if jobs == 1:
        for seed in seeds:
            print(f'=== Training {stem} seed {seed} on {device} ===', flush=True)
            run = run_seed(seed, args.epochs, args.batch, device, project, args.model, args.resume)
            runs.append(run)
            print(json.dumps(run['metrics'], indent=2, default=float), flush=True)
    else:
        # Separate spawned processes avoid sharing random state or accelerator contexts.
        # Each seed has its own run directory and per-run summary; only the parent writes
        # the combined summary. Training wall times include contention between jobs.
        print(f'Training {len(seeds)} independent seeds with {jobs} processes on {device}', flush=True)
        with concurrent.futures.ProcessPoolExecutor(
                max_workers=jobs, mp_context=multiprocessing.get_context('spawn')) as pool:
            futures = {seed: pool.submit(run_seed, seed, args.epochs, args.batch,
                                        device, project, args.model, args.resume) for seed in seeds}
            for seed in seeds:
                run = futures[seed].result()
                runs.append(run)
                print(f'Completed {stem} seed {seed}: {run["metrics"]["mAP50-95"]}', flush=True)
    summary = {
        'generated_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'data': DATA.relative_to(ROOT).as_posix(),
        'data_manifest_sha256': hashlib.sha256((ROOT / 'reports/dataset/split_manifest.csv').read_bytes()).hexdigest(),
        'pretrained_model': args.model,
        'pretrained_checkpoint_sha256': hashlib.sha256(Path(args.model).read_bytes()).hexdigest() if Path(args.model).is_file() else None,
        'device': device,
        'training_jobs': jobs,
        'training_time_note': 'Wall time per seed; concurrent jobs share the device.' if jobs > 1 else 'Sequential seed runs.',
        'versions': {'python': platform.python_version(), 'torch': torch.__version__,
                     'ultralytics': ultralytics.__version__, 'platform': platform.platform(),
                     'hardware': hardware_name(device)},
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
