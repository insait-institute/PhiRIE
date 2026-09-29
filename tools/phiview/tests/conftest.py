"""CPU checks run everywhere; backend and GPU evidence have explicit prerequisites."""
import importlib.util
import pytest


def pytest_addoption(parser):
    parser.addoption('--run-gpu', action='store_true', help='Enable GPU/OpenGL tests on an allocated GPU')


def pytest_configure(config):
    config.addinivalue_line('markers', 'gpu: requires an allocated GPU and configured OpenGL')
    config.addinivalue_line('markers', 'backend: requires the external SimAny/PhiRoom source checkout')


def pytest_collection_modifyitems(config, items):
    from physicalview.config import load_config
    backend = (load_config().repo_root / 'agents').is_dir()
    for item in items:
        if item.get_closest_marker('gpu') and not config.getoption('--run-gpu'):
            item.add_marker(pytest.mark.skip(reason='GPU validation requires --run-gpu'))
        if item.get_closest_marker('backend') and not backend:
            item.add_marker(pytest.mark.skip(reason='External SimAny/PhiRoom checkout is not configured'))
