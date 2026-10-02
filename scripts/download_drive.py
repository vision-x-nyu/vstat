#!/usr/bin/env python3
"""Download the private VSTAT media release using an authenticated rclone remote."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import subprocess
import sys
import time

FOLDER_ID = '1IAUiBXHhqCP_jXM9wZe42dEbIdd-pWoZ'
ACCESS_URL = 'https://drive.google.com/drive/folders/' + FOLDER_ID
SETUP_URL = 'https://github.com/vision-x-nyu/vstat/blob/main/docs/google_drive.md'
DATASET_REVISION = '21b3198fb46627acea9770216a36651a407394b8'
QA_SHA256 = 'a4df882c2872f54d4fa0ec58f2678a8ea0ae2c9f1a0a022e9817834fee74f59c'
EXPECTED_QA = 1500
EXPECTED_VIDEOS = 834


class DownloadError(Exception):
    def __init__(self, status, message, exit_code):
        super().__init__(message)
        self.status = status
        self.exit_code = exit_code


def remote_name(remote):
    name = remote.removesuffix(':')
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', name):
        raise ValueError('--remote must be a configured rclone remote name, for example vstat_drive.')
    return name


def remote_error(stderr):
    """Classify common failures without exposing credentials or raw provider logs."""
    text = stderr.lower()
    if any(word in text for word in ('invalid_grant', 'invalid_client', 'unauthorized_client',
                                     'token expired', 'empty token', '401', 'unauthenticated')):
        return DownloadError('needs_auth', 'Reconnect your rclone remote with the approved Google account. See ' + SETUP_URL, 3)
    if any(word in text for word in ('ratelimit', 'rate limit', 'quota', '429', 'dailylimit')):
        return DownloadError('service_error', 'Google Drive is rate limiting downloads. Retry later.', 5)
    if any(word in text for word in ('403', '404', 'not found', "couldn't find root", 'permission', 'access denied')):
        return DownloadError('access_required', 'Open the access URL, sign in and click Request access; use that same account in rclone. If already approved, check the account and OAuth scope.', 4)
    return DownloadError('service_error', 'Cannot reach the private release. Check your network and rclone setup, then retry.', 5)


def check_access(remote):
    name = remote_name(remote)
    if not shutil.which('rclone'):
        raise DownloadError('needs_setup', 'Install rclone and configure a Google Drive remote. See ' + SETUP_URL, 2)
    remotes = subprocess.run(['rclone', 'listremotes'], capture_output=True, text=True, timeout=15)
    if remotes.returncode or name + ':' not in remotes.stdout.splitlines():
        raise DownloadError('needs_setup', f'Configure the {name} remote with rclone config. See ' + SETUP_URL, 2)
    try:
        result = subprocess.run([
            'rclone', 'cat', name + ':manifest.json', '--drive-root-folder-id', FOLDER_ID,
            '--count', '1048576', '--retries', '1', '--low-level-retries', '2',
            '--contimeout', '10s', '--timeout', '20s'],
            capture_output=True, text=True, timeout=45)
    except subprocess.TimeoutExpired as exc:
        raise DownloadError('service_error', 'Drive access check timed out. Check the network and retry.', 5) from exc
    if result.returncode:
        raise remote_error(result.stderr)
    try:
        manifest = json.loads(result.stdout)
        qa = [entry for entry in manifest['files'] if entry['path'] == 'vstat_qa_clean.json']
        if (manifest.get('schema_version') != 1 or manifest.get('dataset_revision') != DATASET_REVISION
                or len(qa) != 1 or qa[0].get('sha256') != QA_SHA256):
            raise ValueError('Unexpected release manifest.')
    except (ValueError, KeyError, TypeError) as exc:
        raise DownloadError('verification_failed', 'The remote manifest does not match the supported VSTAT release.', 6) from exc


def wait_for_access(remote, seconds):
    deadline = time.monotonic() + seconds
    while True:
        try:
            check_access(remote)
            return
        except DownloadError as exc:
            remaining = deadline - time.monotonic()
            if exc.status != 'access_required' or remaining <= 0:
                raise
            time.sleep(min(30, remaining))


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def local_path(root, relative):
    """Reject absolute paths, traversal and symlinks escaping the dataset root."""
    path = PurePosixPath(relative)
    if not relative or path.is_absolute() or '..' in path.parts or '\\' in relative:
        raise ValueError(f'Invalid release path: {relative!r}')
    target = root.joinpath(*path.parts)
    try:
        target.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f'Release path escapes the dataset root: {relative!r}') from exc
    return target


def verify_release(root, workers=4, decode=False):
    manifest_path = root / 'manifest.json'
    if not manifest_path.is_file():
        raise ValueError(f'Missing {manifest_path}. Download the complete release folder.')
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('schema_version') != 1 or manifest.get('dataset_revision') != DATASET_REVISION:
        raise ValueError('Unexpected release manifest or dataset revision.')
    qa_path = root / 'vstat_qa_clean.json'
    if not qa_path.is_file() or sha256(qa_path) != QA_SHA256:
        raise ValueError('QA file does not match the pinned official dataset revision.')
    qa = json.loads(qa_path.read_text())
    rows = [row for group in qa['data'].values() for row in group]
    paths = {row['video_path'] for row in rows}
    if len(rows) != EXPECTED_QA or len(paths) != EXPECTED_VIDEOS:
        raise ValueError(f'Expected 1500 questions / 834 clips, got {len(rows)} / {len(paths)}.')
    files = manifest['files']
    names = [entry['path'] for entry in files]
    if len(names) != len(set(names)):
        raise ValueError('Duplicate paths in the manifest.')
    media = {name for name in names if name.startswith('videos/')}
    if media != paths:
        raise ValueError(f'Manifest / QA mismatch: {len(paths-media)} missing, {len(media-paths)} extra videos.')
    if 'vstat_qa_clean.json' not in names:
        raise ValueError('QA file is not listed in the manifest.')
    for entry in files:
        local_path(root, entry['path'])
        if not isinstance(entry['bytes'], int) or entry['bytes'] < 0:
            raise ValueError(f'Invalid file size: {entry["path"]}')
        if not re.fullmatch(r'[0-9a-f]{64}', entry['sha256']):
            raise ValueError(f'Invalid SHA-256: {entry["path"]}')

    def check(entry):
        path = local_path(root, entry['path'])
        if not path.is_file():
            return f'MISSING {entry["path"]}'
        if path.stat().st_size != entry['bytes']:
            return f'SIZE MISMATCH {entry["path"]}'
        if sha256(path) != entry['sha256']:
            return f'SHA256 MISMATCH {entry["path"]}'
        return None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        errors = [error for error in pool.map(check, files) if error]
    if errors:
        raise ValueError('\n'.join(errors[:30]) + (f'\n({len(errors)} files failed)' if len(errors) > 30 else ''))
    if decode:
        if not shutil.which('ffmpeg'):
            raise ValueError('--decode requires ffmpeg on PATH.')

        def decode_one(relative):
            proc = subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-xerror', '-threads', '1',
                                   '-i', str(local_path(root, relative)), '-map', '0:v:0', '-an',
                                   '-f', 'null', '-'], capture_output=True, timeout=600)
            return relative if proc.returncode or proc.stderr.strip() else None

        with ThreadPoolExecutor(max_workers=workers) as pool:
            failures = [path for path in pool.map(decode_one, sorted(paths)) if path]
        if failures:
            raise ValueError('Video decode failed: ' + ', '.join(failures))
    return {'questions': len(rows), 'clips': len(paths), 'verified_files': len(files),
            'dataset_revision': DATASET_REVISION, 'decoded': decode}


def copy_command(remote, output, transfers):
    name = remote_name(remote)
    return ['rclone', 'copy', name + ':', str(output), '--drive-root-folder-id', FOLDER_ID,
            '--checksum', '--fast-list', '--transfers', str(transfers), '--checkers', '16',
            '--retries', '5', '--low-level-retries', '10', '--progress']


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('data/vstat'))
    parser.add_argument('--remote', default='vstat_drive', help='An rclone Google Drive remote authenticated as an approved reader.')
    parser.add_argument('--transfers', type=int, default=8)
    parser.add_argument('--workers', type=int, default=4, help='Parallel checksum/decode workers.')
    parser.add_argument('--verify-only', action='store_true', help='Validate an existing local copy without Drive access.')
    parser.add_argument('--decode', action='store_true', help='Additionally decode every video with ffmpeg.')
    parser.add_argument('--dry-run', action='store_true', help='Print the download command without downloading or verifying.')
    parser.add_argument('--check-access', action='store_true', help='Check setup and folder access without downloading media.')
    parser.add_argument('--wait-for-access', type=int, default=0, metavar='SECONDS', help='Wait up to this many seconds for a pending access request; checks every 30 seconds.')
    parser.add_argument('--json', action='store_true', help='Print one final JSON result to stdout; transfer progress goes to stderr.')
    args = parser.parse_args(argv)
    if args.transfers < 1 or args.workers < 1 or args.wait_for_access < 0:
        parser.error('--transfers and --workers must be positive; --wait-for-access must be nonnegative.')
    if sum([args.verify_only, args.dry_run, args.check_access]) > 1:
        parser.error('--verify-only, --dry-run and --check-access are mutually exclusive.')

    def report(status, message, **details):
        result = dict(status=status, message=message, access_url=ACCESS_URL, setup_url=SETUP_URL, **details)
        if args.json:
            print(json.dumps(result))
        else:
            print(message)
            if status in ('access_required', 'needs_auth', 'needs_setup'):
                print('Request access: ' + ACCESS_URL)

    try:
        output = args.output.expanduser().resolve()
        if not args.verify_only:
            cmd = copy_command(args.remote, output, args.transfers)
            if args.dry_run:
                report('dry_run', shlex.join(cmd), command=cmd)
                return 0
            wait_for_access(args.remote, args.wait_for_access)
            if args.check_access:
                report('ready', 'Google Drive access is ready; no media downloaded.')
                return 0
            output.mkdir(parents=True, exist_ok=True)
            print('Downloading private VSTAT release...', file=sys.stderr, flush=True)
            proc = subprocess.run(cmd, stdout=sys.stderr)
            if proc.returncode:
                raise DownloadError('download_failed', 'Download did not finish. Rerun --check-access, then retry; completed matching files are reused.', 5)
        try:
            verified = verify_release(output, args.workers, args.decode)
        except (ValueError, KeyError, TypeError, OSError, subprocess.SubprocessError) as exc:
            raise DownloadError('verification_failed', str(exc), 6) from exc
        report('verified', f'OK: {verified["questions"]} questions, {verified["clips"]} clips; all SHA-256 checks passed.', **verified)
    except DownloadError as exc:
        report(exc.status, str(exc))
        return exc.exit_code
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        report('setup_error', str(exc))
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
