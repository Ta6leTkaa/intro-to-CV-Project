"""Web demo routes must expose saved evidence and reject invalid requests."""
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from demo import server


class DemoTests(unittest.TestCase):
    def handler(self, path):
        handler = object.__new__(server.Handler)
        handler.path = path
        handler.headers = {}
        handler.send = Mock()
        return handler

    def test_status_includes_bundled_examples_without_ml_dependencies(self):
        handler = self.handler('/api/status')
        with patch.object(server, 'dependencies', return_value=['torch']):
            handler.do_GET()
        status, body = handler.send.call_args.args
        self.assertEqual(status, 200)
        self.assertFalse(body['ready'])
        self.assertTrue(set(server.BUNDLED_EXAMPLES).issubset(body['examples']))

    def test_report_uses_committed_metrics(self):
        handler = self.handler('/api/report')
        handler.do_GET()
        status, body = handler.send.call_args.args
        self.assertEqual(status, 200)
        self.assertEqual({r['model'] for r in body['models']}, {'yolo11n', 'yolo11s'})
        self.assertEqual((body['images'], body['seeds']), (78, 3))

    def test_static_file_cannot_escape_its_root(self):
        handler = self.handler('/')
        handler.file(server.STATIC, '../requirements.txt')
        self.assertEqual(handler.send.call_args.args[0], 404)

    def test_invalid_origin_and_nonfinite_threshold_do_not_run_inference(self):
        handler = self.handler('/api/predict')
        handler.headers = {'Origin': 'http://other.invalid', 'Host': '127.0.0.1:8000'}
        with patch.object(server, 'infer') as infer:
            handler.do_POST()
            self.assertEqual(handler.send.call_args.args[0], 403)
            handler.path = '/api/predict?yolo11n=nan'
            handler.headers = {'Content-Length': '10'}
            handler.do_POST()
            self.assertEqual(handler.send.call_args.args[0], 400)
            infer.assert_not_called()


if __name__ == '__main__':
    unittest.main()
