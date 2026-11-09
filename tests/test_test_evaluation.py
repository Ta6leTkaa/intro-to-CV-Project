"""Exercise the metric-only evaluation entry point with synthetic checkpoints/data."""
import contextlib
import csv
import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import evaluate_test as evaluator


class EvaluationTests(unittest.TestCase):
    def test_default_run_preserves_recorded_latency(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); out=root/'reports/results_checkpoint';out.mkdir(parents=True)
            dataset=root/'reports/dataset';dataset.mkdir()
            for name in ['split_manifest.csv','occlusion_review.csv']:(dataset/name).write_text('fixture')
            runs=[]
            for model in evaluator.MODELS:
                for seed in evaluator.SEEDS:
                    p=root/f'reports/training/{model}_seed{seed}'
                    (p/'weights').mkdir(parents=True)
                    (p/'weights/best.pt').write_bytes(f'{model}-{seed}'.encode())
                    digest=hashlib.sha256((p/'weights/best.pt').read_bytes()).hexdigest()
                    runs.append(dict(model=model,seed=seed,checkpoint_sha256=digest))
                    with (p/'results.csv').open('w') as f:
                        w=csv.writer(f);w.writerow(['epoch']);w.writerows([[i] for i in range(1,51)])
            val=root/'reports/evaluation';val.mkdir()
            (val/'validation_summary.json').write_text(json.dumps(dict(split='val',test_used=False,
                selected_thresholds={'yolo11n':.2,'yolo11s':.3},runs=runs)))
            latency={m:dict(median_ms=10,device='mps') for m in evaluator.MODELS}
            (out/'test_summary.json').write_text(json.dumps(dict(split='test',dataset_sha256='data',runs=runs,
                latency=latency,device='mps',versions={'hardware':'original'},generated_utc='old')))
            record=dict(image='fixture.jpg',targets=[[0,0,10,10,0]],predictions=[[0,0,10,10,.9,0]])
            cached=dict(images=[record],ap=dict(map50=1.,map50_95=1.,per_class=[
                {'Class':'helmet','mAP50':1.,'mAP50-95':1.}]))
            model=MagicMock();model.model.parameters.return_value=[]
            with patch.multiple(evaluator,ROOT=root,OUT=out), \
                 patch.object(sys,'argv',['evaluate_test.py','--device','cpu']), \
                 patch.object(evaluator,'dataset',return_value=(['helmet'],[record],'data')), \
                 patch.object(evaluator,'annotations',return_value=[]), \
                 patch.object(evaluator,'predict_checkpoint',return_value=cached), \
                 patch('ultralytics.YOLO',return_value=model), \
                 patch.object(evaluator,'benchmark') as benchmark,contextlib.redirect_stdout(io.StringIO()):
                evaluator.main()
                benchmark.assert_not_called()
            summary=json.loads((out/'test_summary.json').read_text())
            self.assertEqual(summary['latency'],latency)
            self.assertEqual(summary['latency_context']['device'],'mps')
            self.assertEqual(summary['latency_context']['versions'],{'hardware':'original'})
            self.assertEqual(len(summary['runs']),6)


if __name__ == '__main__': unittest.main()
