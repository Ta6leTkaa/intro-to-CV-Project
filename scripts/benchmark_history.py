"""Retain measured latency with its original provenance across metric-only evaluations."""
import json


def retain_benchmark(path, dataset_hash, checkpoints, split):
    if not path.exists():
        return {}, {}
    previous = json.loads(path.read_text())
    if previous.get('dataset_sha256') != dataset_hash or previous.get('split') != split:
        return {}, {}
    context = previous.get('latency_context')
    if context is None:
        # Older summaries benchmarked the first requested seed for each model.
        hashes = {}
        for run in previous.get('runs', []):
            hashes.setdefault(run['model'], run['checkpoint_sha256'])
        context = dict(generated_utc=previous.get('generated_utc'),
                       versions=previous.get('versions'), device=previous.get('device'),
                       checkpoint_sha256=hashes, dataset_sha256=dataset_hash, split=split)
    latency = {name: value for name, value in previous.get('latency', {}).items()
               if name in checkpoints and context.get('checkpoint_sha256', {}).get(name) == checkpoints[name]}
    if not latency:
        return {}, {}
    # Do not relabel a historical timing as measured in the current evaluation environment.
    context = dict(context, checkpoint_sha256={name: checkpoints[name] for name in latency})
    return latency, context


def benchmark_context(dataset_hash, checkpoints, split, versions, device, generated_utc):
    return dict(dataset_sha256=dataset_hash, checkpoint_sha256=checkpoints, split=split,
                versions=versions, device=device, generated_utc=generated_utc)
