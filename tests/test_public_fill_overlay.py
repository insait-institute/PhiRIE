import importlib
from pathlib import Path
import types

import pytest

from agents.edit import inpaint_fill as fill
from run.icra2027 import e3_trellis_generation_pilot as runtime


NAMES = ('pydantic', 'pydantic_core', 'pydantic_core._pydantic_core',
         'annotated_types', 'typing_extensions', 'typing_inspection')


@pytest.fixture
def overlay(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, 'runtime_identity', lambda _: ({}, 'base-packages'))
    monkeypatch.setattr(runtime, 'targeted_runtime_identity', lambda *a, **k: {'base': True})
    root = tmp_path/'overlay'; root.mkdir()
    modules = {}
    for name in NAMES:
        origin = root/(name+'.py'); origin.write_text(name)
        modules[name] = types.SimpleNamespace(__file__=str(origin))
    original = importlib.import_module
    monkeypatch.setattr(importlib, 'import_module', lambda name, *a, **k:
        modules[name] if name in modules else original(name, *a, **k))
    monkeypatch.setenv('SIMANY_FILL_DEPENDENCY_OVERLAY', str(root))
    return root, modules


def test_no_overlay_keeps_existing_runtime_contract(overlay, monkeypatch):
    monkeypatch.delenv('SIMANY_FILL_DEPENDENCY_OVERLAY')
    assert fill.public_runtime('python') == {
        'packages_sha256': 'base-packages', 'targeted_bytes': {'base': True}}


def test_overlay_authenticates_actual_imports_and_native_bytes(overlay):
    root, modules = overlay
    result = fill.public_runtime('python')['dependency_overlay']
    assert result['path'] == str(root)
    assert set(result['imports']) == set(NAMES)
    assert all(Path(row['path']).is_relative_to(root) for row in result['imports'].values())
    native = Path(modules['pydantic_core._pydantic_core'].__file__)
    native.write_text('changed binary')
    changed = fill.public_runtime('python')['dependency_overlay']
    assert changed['tree_sha256'] != result['tree_sha256']
    assert changed['imports']['pydantic_core._pydantic_core'] != result['imports']['pydantic_core._pydantic_core']


@pytest.mark.parametrize('name', NAMES)
def test_shadowed_import_outside_overlay_is_rejected(overlay, tmp_path, name):
    _, modules = overlay
    outside = tmp_path/'outside.py'; outside.write_text('shadow')
    modules[name].__file__ = str(outside)
    with pytest.raises(ValueError, match='outside pinned overlay'):
        fill.public_runtime('python')


@pytest.mark.parametrize('kind', ['missing', 'relative', 'symlink'])
def test_missing_or_unsafe_overlay_is_rejected(overlay, monkeypatch, tmp_path, kind):
    root, _ = overlay
    if kind == 'missing': value = str(tmp_path/'missing')
    elif kind == 'relative': value = 'overlay'
    else:
        link = tmp_path/'link'; link.symlink_to(root, target_is_directory=True); value = str(link)
    monkeypatch.setenv('SIMANY_FILL_DEPENDENCY_OVERLAY', value)
    with pytest.raises((ValueError, RuntimeError)):
        fill.public_runtime('python')
