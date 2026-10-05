"""Download the official release; resume chunks and verify a recorded SHA-256."""
import argparse
import concurrent.futures
import hashlib
import json
import re
import time
import urllib.request
import zipfile
from pathlib import Path

URL = 'https://github.com/ultralytics/assets/releases/download/v0.0.0/construction-ppe.zip'
CONFIG = 'https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/cfg/datasets/construction-ppe.yaml'
ROOT = Path(__file__).resolve().parents[1]

def request(url, headers=None):
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers or {}, method='GET'), timeout=120)

def sha256(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def download(workers=6):
    dest = ROOT / 'data/downloads'
    dest.mkdir(parents=True, exist_ok=True)
    archive = dest / 'construction-ppe.zip'
    lock = ROOT / 'data/source.json'
    if not archive.exists():
        with request(URL, {'Range': 'bytes=0-0'}) as probe:
            resolved = probe.url
            match = re.fullmatch(r'bytes 0-0/(\d+)', probe.headers.get('Content-Range', ''))
            if not match:
                with archive.with_suffix('.partial').open('wb') as f:
                    while chunk := probe.read(1024 * 1024): f.write(chunk)
                archive.with_suffix('.partial').replace(archive)
            else:
                total = int(match[1])
        if match:
            parts = dest / 'parts'
            parts.mkdir(exist_ok=True)
            size = 8 * 1024 * 1024
            ranges = [(i, start, min(start + size, total) - 1) for i, start in enumerate(range(0, total, size))]
            def get_part(item):
                i, start, end = item
                out = parts / f'{i:04d}.part'
                if out.exists() and out.stat().st_size == end - start + 1: return out
                for attempt in range(4):
                    try:
                        with request(resolved, {'Range': f'bytes={start}-{end}'}) as response:
                            if response.status != 206 or response.headers.get('Content-Range') != f'bytes {start}-{end}/{total}':
                                raise ValueError('Server did not honor the byte range')
                            with out.with_suffix('.partial').open('wb') as f:
                                while chunk := response.read(1024 * 1024): f.write(chunk)
                        if out.with_suffix('.partial').stat().st_size != end - start + 1:
                            raise ValueError('Incomplete chunk')
                        out.with_suffix('.partial').replace(out)
                        print(f'Downloaded chunk {i + 1}/{len(ranges)}', flush=True)
                        return out
                    except Exception:
                        if attempt == 3: raise
                        time.sleep(2 * (attempt + 1))
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                files = list(pool.map(get_part, ranges))
            with archive.with_suffix('.partial').open('wb') as f:
                for part in files:
                    with part.open('rb') as src:
                        while chunk := src.read(1024 * 1024): f.write(chunk)
            archive.with_suffix('.partial').replace(archive)
            for part in files: part.unlink()
            parts.rmdir()
    digest = sha256(archive)
    if lock.exists() and json.loads(lock.read_text())['archive_sha256'] != digest:
        raise ValueError('Archive has changed from the locked version; inspect before proceeding')
    config = ROOT / 'data/upstream-construction-ppe.yaml'
    if not config.exists():
        with request(CONFIG) as response: config.write_bytes(response.read())
    with zipfile.ZipFile(archive) as z:
        bad = z.testzip()
        if bad: raise ValueError(f'Corrupt ZIP member: {bad}')
        raw = ROOT / 'data/raw/construction-ppe'
        raw.mkdir(parents=True, exist_ok=True)
        for name in z.namelist():
            target = (raw / name).resolve()
            if not target.is_relative_to(raw.resolve()): raise ValueError('Unsafe archive path')
        z.extractall(raw)
    lock.write_text(json.dumps({'dataset': 'Ultralytics Construction-PPE', 'archive_url': URL,
        'archive_sha256': digest, 'archive_bytes': archive.stat().st_size,
        'config_url': CONFIG, 'config_sha256': sha256(config),
        'license_declared_by_publisher': 'AGPL-3.0',
        'documentation': 'https://docs.ultralytics.com/datasets/detect/construction-ppe'}, indent=2) + '\n')
    print(f'Archive verified and extracted. SHA-256: {digest}', flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    download(args.workers)
