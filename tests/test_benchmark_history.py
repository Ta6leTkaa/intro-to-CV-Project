"""Metric-only reevaluation retains timings without relabeling their environment."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from benchmark_history import retain_benchmark


class BenchmarkHistoryTests(unittest.TestCase):
    def test_legacy_timing_retains_original_environment(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'summary.json'
            prior=dict(split='test', dataset_sha256='data', device='mps', versions={'hardware':'old'},
                       generated_utc='original date', runs=[dict(model='yolo11n', checkpoint_sha256='weights')],
                       latency={'yolo11n':dict(median_ms=7.58,device='mps')})
            path.write_text(json.dumps(prior))
            latency,context=retain_benchmark(path,'data',{'yolo11n':'weights'},'test')
            self.assertEqual(latency,prior['latency'])
            self.assertEqual(context['versions'],{'hardware':'old'})
            self.assertEqual(context['device'],'mps')
            # A second reevaluation retains the original timing provenance, too.
            prior.update(device='cpu',versions={'hardware':'new'},latency_context=context)
            path.write_text(json.dumps(prior))
            self.assertEqual(retain_benchmark(path,'data',{'yolo11n':'weights'},'test'),(latency,context))
            self.assertEqual(retain_benchmark(path,'other',{'yolo11n':'weights'},'test'),({},{}))
            self.assertEqual(retain_benchmark(path,'data',{'yolo11n':'changed'},'test'),({},{}))
            self.assertEqual(retain_benchmark(path,'data',{'yolo11n':'weights'},'val'),({},{}))

    def test_missing_measurement_is_not_fabricated(self):
        with tempfile.TemporaryDirectory() as temp:
            self.assertEqual(retain_benchmark(Path(temp)/'missing.json','data',{},'test'),({},{}))


if __name__ == '__main__': unittest.main()
