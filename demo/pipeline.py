"""Image inference for the local web demo."""
import base64
import io
import math
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = {'yolo11n': 0.20, 'yolo11s': 0.30}
LOADED = {}


def run_detection(payload, selected=None, thresholds=None, device='cpu'):
    """Run seed-42 checkpoints with the web demo settings."""
    selected = list(MODELS) if selected is None else list(selected)
    thresholds = dict(MODELS) if thresholds is None else dict(thresholds)
    if not selected or any(name not in MODELS for name in selected):
        raise ValueError('Select yolo11n and/or yolo11s.')
    if any(name not in thresholds or not math.isfinite(thresholds[name])
           or not 0.01 <= thresholds[name] <= 0.95 for name in selected):
        raise ValueError('Confidence must be between 0.01 and 0.95.')
    from PIL import Image, ImageOps, UnidentifiedImageError
    from ultralytics import YOLO
    import torch
    import sys
    sys.path.insert(0, str(ROOT / 'scripts'))
    from mps_compat import configure_mps_nms
    configure_mps_nms(device)
    torch.set_num_threads(4)
    try:
        with Image.open(io.BytesIO(payload)) as raw:
            if raw.width * raw.height > 24_000_000:
                raise ValueError('Maximum image size is 24 megapixels.')
            image = ImageOps.exif_transpose(raw).convert('RGB')
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise ValueError('Cannot read image. Use JPEG, PNG or WebP.') from error
    output = io.BytesIO()
    image.save(output, format='JPEG', quality=93)
    results = []
    for name in selected:
        checkpoint = ROOT / f'reports/training/{name}_seed42/weights/best.pt'
        if not checkpoint.is_file():
            raise ValueError(f'Missing checkpoint: {checkpoint.relative_to(ROOT)}')
        cold = (device, name) not in LOADED
        if cold:
            LOADED[device, name] = YOLO(str(checkpoint))
        start = time.perf_counter()
        prediction = LOADED[device, name].predict(image, device=device, imgsz=640, rect=False,
                                         conf=thresholds[name], iou=0.70, max_det=300,
                                         quantize=32, verbose=False, save=False)[0]
        elapsed = (time.perf_counter() - start) * 1000
        boxes = [dict(xyxy=b[:4], confidence=b[4], class_id=int(b[5]),
                      label=prediction.names[int(b[5])])
                 for b in prediction.boxes.data.cpu().tolist()]
        results.append(dict(model=name, confidence=thresholds[name], boxes=boxes,
                            elapsed_ms=round(elapsed, 1), cold_start=cold))
    return dict(image='data:image/jpeg;base64,' + base64.b64encode(output.getvalue()).decode(),
                width=image.width, height=image.height, results=results, device=device)
