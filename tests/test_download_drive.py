import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location('download_drive', Path(__file__).resolve().parents[1] / 'scripts/download_drive.py')
download = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(download)


class ReleaseVerificationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.clip = self.root / 'videos/synthetic/demo/001.mp4'
        self.clip.parent.mkdir(parents=True)
        self.clip.write_bytes(b'synthetic-test-video')
        self.qa = self.root / 'vstat_qa_clean.json'
        self.qa.write_text(json.dumps({'data': {'demo': [
            {'video_path': 'videos/synthetic/demo/001.mp4'},
            {'video_path': 'videos/synthetic/demo/001.mp4'}]}}))
        self.files = [self.record(self.clip), self.record(self.qa)]
        self.write_manifest()
        for name, value in [('EXPECTED_QA', 2), ('EXPECTED_VIDEOS', 1), ('QA_SHA256', download.sha256(self.qa))]:
            p = patch.object(download, name, value)
            p.start()
            self.addCleanup(p.stop)

    def record(self, path):
        return {'path': path.relative_to(self.root).as_posix(), 'bytes': path.stat().st_size, 'sha256': download.sha256(path)}

    def write_manifest(self):
        (self.root / 'manifest.json').write_text(json.dumps({'schema_version': 1,
            'dataset_revision': download.DATASET_REVISION, 'files': self.files}))

    def verify(self):
        with contextlib.redirect_stdout(io.StringIO()):
            download.verify_release(self.root)

    def test_valid_release_with_multiple_questions_per_clip(self):
        self.verify()

    def test_same_size_corruption_is_rejected(self):
        self.clip.write_bytes(b'x' * self.clip.stat().st_size)
        with self.assertRaisesRegex(ValueError, 'SHA256 MISMATCH'):
            self.verify()

    def test_missing_clip_is_rejected(self):
        self.clip.unlink()
        with self.assertRaisesRegex(ValueError, 'MISSING'):
            self.verify()

    def test_omitted_manifest_video_is_rejected(self):
        self.files = [self.record(self.qa)]
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, 'Manifest / QA mismatch'):
            self.verify()

    def test_metadata_cannot_escape_release_directory(self):
        self.files.append({'path': '../outside.txt', 'bytes': 0, 'sha256': '0' * 64})
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, 'Invalid release path'):
            self.verify()

    def test_symlink_cannot_escape_release_directory(self):
        with tempfile.TemporaryDirectory() as outside:
            (self.root / 'outside').symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, 'escapes'):
                download.local_path(self.root, 'outside/secret')

    def test_changed_qa_revision_is_rejected(self):
        self.qa.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'pinned official'):
            self.verify()

    def test_failed_download_never_reports_verification_success(self):
        with patch.object(download, 'wait_for_access'), \
             patch.object(download.subprocess, 'run') as run, \
             patch.object(download, 'verify_release') as verify, \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            run.return_value.returncode = 3
            self.assertEqual(download.main(['--output', str(self.root)]), 5)
            verify.assert_not_called()

    def test_download_is_non_destructive_and_folder_scoped(self):
        command = download.copy_command('my_drive:', Path('/tmp/path with spaces'), 4)
        self.assertEqual(command[:4], ['rclone', 'copy', 'my_drive:', '/tmp/path with spaces'])
        self.assertIn(download.FOLDER_ID, command)
        self.assertNotIn('sync', command)
        with self.assertRaises(ValueError):
            download.copy_command('other:path', self.root, 1)

    def test_json_verification_is_one_result(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = download.main(['--output', str(self.root), '--verify-only', '--json'])
        result = json.loads(output.getvalue())
        self.assertEqual(code, 0)
        self.assertEqual(result['status'], 'verified')
        self.assertEqual((result['questions'], result['clips']), (2, 1))


class AccessTests(unittest.TestCase):
    def run_check(self, responses=None, installed=True):
        output = io.StringIO()
        with patch.object(download.shutil, 'which', return_value='/usr/bin/rclone' if installed else None), \
             patch.object(download.subprocess, 'run', side_effect=responses) as run, \
             contextlib.redirect_stdout(output):
            code = download.main(['--check-access', '--json'])
        return code, json.loads(output.getvalue()), run

    def response(self, stdout='', stderr='', returncode=0):
        return SimpleNamespace(stdout=stdout, stderr=stderr, returncode=returncode)

    def test_missing_dependency_is_setup_not_access(self):
        code, result, run = self.run_check(installed=False)
        self.assertEqual((code, result['status']), (2, 'needs_setup'))
        run.assert_not_called()

    def test_unconfigured_remote_is_setup_not_access(self):
        code, result, run = self.run_check([self.response('other_drive:\n')])
        self.assertEqual((code, result['status']), (2, 'needs_setup'))
        self.assertEqual(run.call_count, 1)

    def test_provider_failures_require_different_actions(self):
        cases = [
            ('oauth2: invalid_grant', 3, 'needs_auth'),
            ('googleapi: Error 404: File not found', 4, 'access_required'),
            ('googleapi: Error 403: rateLimitExceeded', 5, 'service_error'),
            ('connection refused', 5, 'service_error'),
        ]
        for stderr, expected_code, expected_status in cases:
            with self.subTest(stderr=stderr):
                code, result, _ = self.run_check([
                    self.response('vstat_drive:\n'), self.response(stderr=stderr, returncode=1)])
                self.assertEqual((code, result['status']), (expected_code, expected_status))
                self.assertEqual(result['access_url'], download.ACCESS_URL)

    def test_readable_wrong_release_is_not_ready(self):
        code, result, _ = self.run_check([
            self.response('vstat_drive:\n'), self.response('{}')])
        self.assertEqual((code, result['status']), (6, 'verification_failed'))

    def test_ready_check_does_not_download_media(self):
        manifest = dict(schema_version=1, dataset_revision=download.DATASET_REVISION,
                        files=[dict(path='vstat_qa_clean.json', sha256=download.QA_SHA256)])
        code, result, run = self.run_check([
            self.response('vstat_drive:\n'), self.response(json.dumps(manifest))])
        self.assertEqual((code, result['status']), (0, 'ready'))
        self.assertEqual(run.call_count, 2)
        self.assertEqual(run.call_args.args[0][1], 'cat')

    def test_wait_continues_after_access_is_granted(self):
        denied = download.DownloadError('access_required', 'pending', 4)
        with patch.object(download, 'check_access', side_effect=[denied, None]) as check, \
             patch.object(download.time, 'monotonic', side_effect=[0, 0]), \
             patch.object(download.time, 'sleep') as sleep:
            download.wait_for_access('vstat_drive', 30)
        self.assertEqual(check.call_count, 2)
        sleep.assert_called_once_with(30)

    def test_wait_stops_at_deadline(self):
        denied = download.DownloadError('access_required', 'pending', 4)
        with patch.object(download, 'check_access', side_effect=denied), \
             patch.object(download.time, 'monotonic', side_effect=[0, 30]), \
             patch.object(download.time, 'sleep') as sleep:
            with self.assertRaises(download.DownloadError):
                download.wait_for_access('vstat_drive', 30)
        sleep.assert_not_called()

    def test_wait_does_not_retry_expired_credentials(self):
        auth = download.DownloadError('needs_auth', 'reconnect', 3)
        with patch.object(download, 'check_access', side_effect=auth), \
             patch.object(download.time, 'sleep') as sleep:
            with self.assertRaises(download.DownloadError):
                download.wait_for_access('vstat_drive', 600)
        sleep.assert_not_called()


if __name__ == '__main__':
    unittest.main()
