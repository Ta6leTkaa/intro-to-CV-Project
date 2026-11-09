"""Training reruns must not overwrite retained scientific artifacts."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from train_yolo11n import run_seed, validate_training_destination


class TrainingOutputTests(unittest.TestCase):
    def test_existing_run_is_rejected_before_training_or_failure_log(self):
        with tempfile.TemporaryDirectory() as temp:
            project = Path(temp)
            run = project/'yolo11n_seed42'
            run.mkdir(); (run/'results.csv').write_text('retained epoch logs')
            with patch('train_yolo11n._run_seed') as train:
                with self.assertRaises(FileExistsError):
                    run_seed(42, 2, 16, 'cpu', project, 'yolo11n.pt')
                train.assert_not_called()
            self.assertEqual((run/'results.csv').read_text(), 'retained epoch logs')
            self.assertFalse((run/'failure.log').exists())

    def test_summary_and_later_seed_are_protected(self):
        with tempfile.TemporaryDirectory() as temp:
            project=Path(temp)
            summary=project/'yolo11n_summary.json'; summary.write_text('{}')
            with self.assertRaises(FileExistsError):
                validate_training_destination(project, 'yolo11n.pt', [45])
            summary.unlink()
            run=project/'yolo11n_seed44'; run.mkdir(); (run/'best.pt').write_bytes(b'weights')
            with self.assertRaises(FileExistsError):
                validate_training_destination(project, 'yolo11n.pt', [42,43,44])
            self.assertFalse((project/'yolo11n_seed42').exists())
            validate_training_destination(project, 'yolo11n.pt', [44], resume=True)
            validate_training_destination(project/'fresh', 'yolo11n.pt', [42])


if __name__ == '__main__': unittest.main()
