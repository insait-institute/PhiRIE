"""Immutable, credential-free task receipts and artifact identities."""
from __future__ import annotations
import hashlib
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load(path):
    path = Path(path)
    if path.suffix in ('.yaml', '.yml'):
        import yaml
        return yaml.safe_load(path.read_text())
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def save(path, value, *, jsonl=False):
    """Atomic publication without overwriting an existing result."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = b'\n'.join(canonical(v) for v in value) if jsonl else canonical(value)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix='.writing-')
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(body + b'\n'); f.flush(); os.fsync(f.fileno())
        os.link(name, path)  # atomic and fails if destination already exists
    finally:
        os.unlink(name)


def checked_path(record):
    p = Path(record['path']).expanduser().absolute()
    if not p.is_file() or sha(p) != record['sha256']:
        raise ValueError(f'artifact identity mismatch: {p}')
    return p


def receipt(path):
    p = Path(path).absolute()
    return {'path': str(p), 'sha256': sha(p), 'bytes': p.stat().st_size}


def source_identity(root):
    root = Path(root).absolute()
    head = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain', '--untracked-files=no'], text=True).strip()
    if dirty:
        raise ValueError('tracked source is dirty; commit before sealing a task')
    return {'root': str(root), 'commit': head}


def local_model(config):
    """No implicit latest downloads during a frozen experiment."""
    path = Path(config['checkpoint']).expanduser().absolute()
    if not path.exists():
        raise FileNotFoundError(path)
    manifest = load(config['checkpoint_manifest'])
    if not manifest.get('files'):
        raise ValueError('checkpoint manifest must list actual files')
    if path.is_file() and set(manifest['files']) != {path.name}:
        raise ValueError('single-file checkpoint manifest must name that exact file')
    for rel, checksum in manifest['files'].items():
        q = Path(rel)
        if q.is_absolute() or '..' in q.parts:
            raise ValueError('unsafe checkpoint manifest path')
        if sha(path / q if path.is_dir() else path) != checksum:
            raise ValueError(f'checkpoint changed: {rel}')
    return str(path)


def safe_id(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_.-]+', value):
        raise ValueError('ID must contain only letters, digits, _, . or -')
    return value
