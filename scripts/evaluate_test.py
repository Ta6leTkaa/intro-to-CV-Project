"""Results checkpoint: held-out test evaluation with Stage 4 settings frozen first."""
import argparse
import csv
import json
import platform
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from benchmark_history import benchmark_context, retain_benchmark
from detection_metrics import evaluate_threshold, match_detections
from evaluate_models import (ROOT, SETTINGS, benchmark, dataset, digest, mean_std,
                             predict_checkpoint, write_csv)
from mps_compat import configure_mps_nms, configure_mps_validation
from train_yolo11n import hardware_name, pick_device

OUT = ROOT / 'reports/results_checkpoint'
MODELS = ['yolo11n', 'yolo11s']
SEEDS = [42, 43, 44]


def annotations(records, names):
    """Join original line-number visibility tags to prepared object order; verify class/box."""
    with (ROOT / 'reports/dataset/split_manifest.csv').open() as stream:
        manifest = list(csv.DictReader(stream))
    with (ROOT / 'reports/dataset/occlusion_review.csv').open() as stream:
        tags = {(r['image'], int(r['line'])): r for r in csv.DictReader(stream)}
    lookup = {f"{r['original_split']}__{Path(r['image']).name}": r
              for r in manifest if r['split'] == 'test'}
    objects = []
    for record in records:
        p = ROOT / record['image']
        row = lookup[p.name]
        source = ROOT / 'data/raw/construction-ppe' / row['label']
        # Raw dataset location follows download_dataset.py.
        if not source.exists():
            raise FileNotFoundError(source)
        raw = [list(map(float, s.split())) for s in source.read_text().splitlines() if s.strip()]
        with Image.open(p) as im:
            w, h = im.size
        used = set()
        for index, target in enumerate(record['targets']):
            x1, y1, x2, y2, cls = target
            normalized = np.array([cls, (x1+x2)/2/w, (y1+y2)/2/h, (x2-x1)/w, (y2-y1)/h])
            candidates = [i for i, box in enumerate(raw) if i not in used and np.allclose(box, normalized, atol=1e-7, rtol=0)]
            if not candidates:
                raise ValueError(f'Cannot join source object {p.name}:{index}')
            line = candidates[0] + 1
            used.add(line-1)
            tag = tags.get((row['image'], line))
            if cls < 5 and (tag is None or tag['class_name'] != names[int(cls)]):
                raise ValueError('Missing or inconsistent PPE visibility review')
            area = (x2-x1)*(y2-y1)*(640/max(w,h))**2
            objects.append(dict(image=record['image'], target_index=index, source_image=row['image'],
                                source_line=line, class_name=names[int(cls)], class_id=int(cls),
                                area_640=area, size='small' if area < 32**2 else 'medium' if area < 96**2 else 'large',
                                visibility=tag['final_tag'] if tag else 'not_reviewed',
                                truncated=tag['truncated'] if tag else 'not_reviewed'))
    if Counter(o['visibility'] for o in objects if o['class_id'] < 5) != dict(clear=259, occluded=72, uncertain=190):
        raise ValueError('Visibility review differs from the frozen 521-object review.')
    return objects


def subgroup_rows(images, objects, confidence, model, seed):
    hits = set()
    for im in images:
        _, pairs = match_detections(im['predictions'], im['targets'], confidence)
        hits.update((im['image'], g) for _, g in pairs)
    rows = []
    for dimension, groups in [('size', ['small','medium','large']),
                              ('visibility', ['clear','occluded','uncertain']),
                              ('truncated', ['yes','no','uncertain'])]:
        for group in groups:
            for cls in ['ALL'] + sorted({o['class_name'] for o in objects}):
                subset = [o for o in objects if o[dimension] == group and (cls == 'ALL' or o['class_name'] == cls)]
                if not subset:
                    continue
                tp = sum((o['image'], o['target_index']) in hits for o in subset)
                rows.append(dict(model=model, seed=seed, dimension=dimension, group=group, class_name=cls,
                                 support=len(subset), tp=tp, fn=len(subset)-tp, recall=tp/len(subset)))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--refresh', action='store_true')
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive')
    import torch
    import ultralytics
    from ultralytics.utils.torch_utils import get_cpu_info
    torch.set_num_threads(args.threads)
    device = pick_device(args.device)
    OUT.mkdir(parents=True, exist_ok=True)
    val_path = ROOT / 'reports/evaluation/validation_summary.json'
    val = json.loads(val_path.read_text())
    if val['split'] != 'val' or val['test_used'] is not False:
        raise ValueError('Threshold selection must use validation only.')
    thresholds = val['selected_thresholds']
    if thresholds != {'yolo11n': 0.2, 'yolo11s': 0.3}:
        raise ValueError('Thresholds differ from the frozen Stage 4 selection.')
    checkpoints = {(m,s): ROOT / f'reports/training/{m}_seed{s}/weights/best.pt' for m in MODELS for s in SEEDS}
    val_hashes = {(r['model'], r['seed']): r['checkpoint_sha256'] for r in val['runs']}
    for key, checkpoint in checkpoints.items():
        if digest(checkpoint) != val_hashes[key]:
            raise ValueError(f'Checkpoint differs from Stage 4: {key}')
        log = checkpoint.parents[1] / 'results.csv'
        with log.open() as stream:
            epochs = [int(r['epoch']) for r in csv.DictReader(stream)]
        if epochs != list(range(1,51)):
            raise ValueError(f'Incomplete or duplicated epoch log: {log}')
    # Persist the selection protocol before reading test images or making predictions.
    protocol = dict(validation_summary_sha256=digest(val_path), thresholds=thresholds,
                    checkpoint_sha256={f'{m}_{s}': val_hashes[m,s] for m,s in checkpoints},
                    settings=SETTINGS, seeds=SEEDS, split='test', matching_iou=0.5,
                    model_selection='validation only; all three seeds reported; no test tuning',
                    benchmark_seed=42, benchmark_confidence=0.25)
    lock = OUT / 'frozen_protocol.json'
    if lock.exists() and json.loads(lock.read_text())['protocol'] != protocol:
        raise ValueError('Frozen test protocol has changed; refusing to overwrite')
    if not lock.exists():
        lock.write_text(json.dumps(dict(frozen_utc=datetime.now(timezone.utc).isoformat(), protocol=protocol), indent=2)+'\n')
    names, records, data_hash = dataset('test')
    objects = annotations(records, names)
    write_csv(OUT / 'test_objects.csv', objects)
    versions = dict(python=platform.python_version(), torch=torch.__version__,
                    ultralytics=ultralytics.__version__, platform=platform.platform(), hardware=hardware_name(device),
                    torch_threads=torch.get_num_threads())
    postprocess = configure_mps_nms(device)
    validation_backend = configure_mps_validation(device)
    runs, classes, subgroups, complexity = [], [], [], []
    for model in MODELS:
        for seed in SEEDS:
            print(f'TEST: {model} seed {seed}', flush=True)
            started = time.perf_counter()
            cached = predict_checkpoint(checkpoints[model,seed], model, seed, device, records, names,
                                        data_hash, versions, args.refresh, split='test', output_dir=OUT)
            fixed = evaluate_threshold(cached['images'], thresholds[model], names)
            reference = evaluate_threshold(cached['images'], 0.25, names)
            run = dict(model=model, seed=seed, checkpoint_sha256=val_hashes[model,seed],
                       map50_95=cached['ap']['map50_95'], map50=cached['ap']['map50'],
                       **{k:fixed[k] for k in ['macro_precision','macro_recall','macro_f1']},
                       macro_f1_ref025=reference['macro_f1'], confidence=thresholds[model],
                       evaluation_wall_seconds=time.perf_counter()-started)
            runs.append(run)
            ap = {r['Class']:r for r in cached['ap']['per_class']}
            for c in fixed['per_class']:
                a = ap.get(c['class_name'], {})
                classes.append(dict(model=model, seed=seed, **c, ap50=a.get('mAP50'), ap50_95=a.get('mAP50-95')))
            subgroups += subgroup_rows(cached['images'], objects, thresholds[model], model, seed)
        from ultralytics import YOLO
        net = YOLO(str(checkpoints[model,42])).model
        complexity.append(dict(model=model, parameters=sum(p.numel() for p in net.parameters()),
                               checkpoint_bytes=checkpoints[model,42].stat().st_size))
    comparison = []
    for model in MODELS:
        rr = [r for r in runs if r['model'] == model]
        row = dict(model=model, seeds=3, confidence=thresholds[model])
        for metric in ['map50_95','map50','macro_precision','macro_recall','macro_f1','macro_f1_ref025']:
            row.update({metric+'_'+k:v for k,v in mean_std([r[metric] for r in rr]).items()})
        comparison.append(row)
    benchmark_hashes = {model: val_hashes[model, 42] for model in MODELS}
    latency, latency_context = retain_benchmark(OUT / 'test_summary.json', data_hash,
                                                benchmark_hashes, 'test')
    if args.benchmark:
        latency = {}
        for model in MODELS:
            latency[model] = benchmark(checkpoints[model,42], records, device, 0.25, 20, 100)
        latency_context = benchmark_context(data_hash, benchmark_hashes, 'test', versions, device,
                                            datetime.now(timezone.utc).isoformat())
    for name, rows in [('per_seed',runs),('model_comparison',comparison),('per_class',classes),
                       ('subgroup_recall',subgroups),('complexity',complexity)]:
        write_csv(OUT / (name+'.csv'), rows)
    write_csv(OUT / 'visibility_size_distribution.csv', [dict(visibility=v,size=s,support=sum(o['visibility']==v and o['size']==s for o in objects)) for v in ['clear','occluded','uncertain'] for s in ['small','medium','large']])
    result = dict(generated_utc=datetime.now(timezone.utc).isoformat(), split='test', images=len(records),
                  objects=len(objects), class_names=names, dataset_sha256=data_hash, versions=versions,
                  device=device, cpu_identification=get_cpu_info(),
                  postprocess_backend=postprocess, validation_backend=validation_backend,
                  protocol=protocol, ap_multilabel=True, threshold_prediction_multilabel=False,
                  runs=runs, comparison=comparison, latency=latency, latency_context=latency_context,
                  complexity=complexity,
                  annotation_hashes={n:digest(ROOT / 'reports/dataset' / n) for n in ['split_manifest.csv','occlusion_review.csv']})
    (OUT/'test_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(comparison=comparison,latency=latency),indent=2))


if __name__ == '__main__':
    main()
