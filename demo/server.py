"""Local PPE demo. Run: python demo/server.py (no extra web dependencies)."""
import argparse
import csv
import importlib.util
import json
import math
import mimetypes
import threading
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from demo.pipeline import MODELS, run_detection

STATIC = ROOT / 'demo'
IMAGES = ROOT / 'data/prepared/construction-ppe/images/test'
BUNDLED_IMAGES = STATIC / 'examples'
BUNDLED_EXAMPLES = {
    'clear_ppe.jpg': 'Clear PPE',
    'multiple_workers.jpg': 'Multiple workers',
    'distant_workers.jpg': 'Distant workers',
}
REPORT = ROOT / 'reports/results_checkpoint'
LIMIT = 15 * 1024 * 1024
LOCK = threading.Lock()
DEVICE = 'cpu'


def dependencies():
    return [name for name in ('torch', 'ultralytics', 'PIL')
            if importlib.util.find_spec(name) is None]


def examples():
    bundled = [name for name in BUNDLED_EXAMPLES if (BUNDLED_IMAGES / name).is_file()]
    prepared = sorted(p.name for p in IMAGES.glob('*')
                      if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png'}
                      and p.name not in BUNDLED_EXAMPLES)
    return bundled + prepared


def infer(payload, selected, thresholds):
    return run_detection(payload, selected, thresholds, device=DEVICE)


class Handler(BaseHTTPRequestHandler):
    def send(self, status, body, content_type='application/json; charset=utf-8'):
        if not isinstance(body, bytes):
            body = json.dumps(body, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        self.wfile.write(body)

    def file(self, root, name):
        path = (root / unquote(name)).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            return self.send(404, {'error': 'File not found'})
        self.send(200, path.read_bytes(), mimetypes.guess_type(path.name)[0] or 'application/octet-stream')

    def do_GET(self):
        path = urlparse(self.path).path
        if path == '/api/status':
            missing = dependencies()
            return self.send(200, dict(ready=not missing, missing=missing, device=DEVICE,
                                       examples=examples(), example_labels=BUNDLED_EXAMPLES,
                                       thresholds=MODELS))
        if path == '/api/report':
            with (REPORT / 'model_comparison.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            return self.send(200, dict(models=rows, images=78, seeds=3))
        if path.startswith('/examples/'):
            name = unquote(path[len('/examples/'):])
            if name in BUNDLED_EXAMPLES:
                return self.file(BUNDLED_IMAGES, name)
            return self.file(IMAGES, path[len('/examples/'):])
        if path.startswith('/figures/'):
            return self.file(REPORT / 'figures', path[len('/figures/'):])
        assets = {'/': 'index.html', '/app.js': 'app.js', '/style.css': 'style.css'}
        if path in assets:
            return self.file(STATIC, assets[path])
        self.send(404, {'error': 'Page not found'})

    def do_POST(self):
        url = urlparse(self.path)
        if url.path != '/api/predict':
            return self.send(404, {'error': 'Route not found'})
        # Browser requests must originate from this server; uploads never go to a cloud.
        origin = self.headers.get('Origin')
        if origin and origin != 'http://' + self.headers.get('Host', ''):
            return self.send(403, {'error': 'Invalid request origin'})
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= LIMIT:
                return self.send(413, {'error': 'Choose an image up to 15 MB.'})
            query = parse_qs(url.query)
            selected = query.get('model', ['both'])[0]
            if selected not in (*MODELS, 'both'):
                raise ValueError('Unknown model')
            thresholds = {name: float(query.get(name, [default])[0]) for name, default in MODELS.items()}
            if any(not math.isfinite(v) or not 0.01 <= v <= 0.95 for v in thresholds.values()):
                raise ValueError('Confidence must be between 0.01 and 0.95')
            missing = dependencies()
            if missing:
                return self.send(503, {'error': 'Install dependencies: python -m pip install -r demo/requirements.txt',
                                       'missing': missing})
            if not LOCK.acquire(blocking=False):
                return self.send(429, {'error': 'Detection is busy. Please try again shortly.'})
            try:
                result = infer(self.rfile.read(length), list(MODELS) if selected == 'both' else [selected], thresholds)
            finally:
                LOCK.release()
            self.send(200, result)
        except ValueError as error:
            self.send(400, {'error': str(error)})
        except Exception:
            import traceback
            traceback.print_exc()
            self.send(500, {'error': 'Detection failed. Check the server terminal.'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default='cpu')
    args = parser.parse_args()
    DEVICE = args.device
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    print(f'PPE demo: http://127.0.0.1:{args.port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
