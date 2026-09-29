"""CPU tests for physicalview.app primitives (EventBus, Selection, Context)."""
from __future__ import annotations

import types

from physicalview.app import Context, EventBus, Selection
from physicalview.config import load_config
from physicalview.gpu import GpuInfo


def test_eventbus_delivers_and_isolates_failures(caplog):
    bus = EventBus()
    seen = []
    bus.subscribe("t", lambda p: seen.append(p))
    bus.subscribe("t", lambda p: 1 / 0)          # broken handler must not break others
    bus.subscribe("t", lambda p: seen.append(p * 2))
    bus.publish("t", 21)
    assert seen == [21, 42]
    assert "event handler for t failed" in caplog.text


def test_eventbus_unsubscribe():
    bus = EventBus()
    seen = []
    off = bus.subscribe("x", seen.append)
    bus.publish("x", 1)
    off()
    bus.publish("x", 2)
    assert seen == [1]


def test_selection_object_names():
    sel = Selection(kind="object", object_ids=[3, 12])
    assert sel.object_names == ["obj_03", "obj_12"]
    assert Selection().kind == "none"


def test_context_status_log_and_selection_events():
    cfg = load_config()
    ctx = Context(server=None, config=cfg, gpu=GpuInfo(present=False), jobs=None)
    handle = types.SimpleNamespace(content="")
    ctx.attach_status(handle)
    ctx.set_status("hello")
    assert "hello" in handle.content and "hello" in ctx.recent_log()
    got = []
    ctx.events.subscribe("selection.changed", got.append)
    ctx.set_selection(Selection(kind="box", box_center=(0, 0, 0), box_size=(1, 1, 1)))
    assert got and got[0].kind == "box"
    got2 = []
    ctx.events.subscribe("scene.loaded", got2.append)
    ctx.set_scene("scene-obj")
    assert ctx.scene == "scene-obj" and got2 == ["scene-obj"]
