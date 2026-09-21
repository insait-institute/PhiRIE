"""Fail-closed client and stream accounting for an external Harmonizer service."""
from __future__ import annotations

import base64
import io
import json
import socket
import struct
import time
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from PIL import Image


class EnhancerError(RuntimeError):
    pass


class EnhancerProtocol(Protocol):
    def reset_stream(self, stream_id: str) -> None: ...
    def enhance(self, image: np.ndarray, *, stream_id: str,
                frame_index: int) -> tuple[np.ndarray, dict]: ...
    def model_info(self) -> dict: ...


@dataclass
class IdentityEnhancer:
    """Deterministic CI backend. It is never labeled Harmonizer in results."""

    streams: dict[str, int] = field(default_factory=dict)

    def reset_stream(self, stream_id: str) -> None:
        self.streams.pop(stream_id, None)

    def enhance(self, image: np.ndarray, *, stream_id: str,
                frame_index: int) -> tuple[np.ndarray, dict]:
        expected = self.streams.get(stream_id, 0)
        if frame_index != expected:
            raise EnhancerError(
                f"stream {stream_id!r} expected frame {expected}, got {frame_index}")
        self.streams[stream_id] = expected + 1
        return np.asarray(image).copy(), {"backend": "identity", "latency_ms": 0.0}

    def model_info(self) -> dict:
        return {"backend": "identity", "scientific_result": False}


def _encode_png(image: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _decode_png(value: str) -> np.ndarray:
    data = base64.b64decode(value.encode("ascii"), validate=True)
    with Image.open(io.BytesIO(data)) as image:
        return np.asarray(image.convert("RGB"))


class SocketHarmonizerClient:
    """Length-prefixed JSON client for ``integrations/harmonizer/server.py``."""

    def __init__(self, socket_path: str, timeout_s: float = 30.0):
        self.socket_path = socket_path
        self.timeout_s = float(timeout_s)
        self._next_frame: dict[str, int] = {}

    def _request(self, payload: dict) -> dict:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                client.settimeout(self.timeout_s)
                client.connect(self.socket_path)
                client.sendall(struct.pack("!Q", len(body)) + body)
                response_size = struct.unpack("!Q", _recv_exact(client, 8))[0]
                response = json.loads(_recv_exact(client, response_size))
        except Exception as exc:
            raise EnhancerError(f"Harmonizer request failed: {type(exc).__name__}: {exc}") from exc
        if not response.get("ok", False):
            raise EnhancerError(response.get("error", "unknown Harmonizer service error"))
        return response

    def reset_stream(self, stream_id: str) -> None:
        self._request({"op": "reset_stream", "stream_id": stream_id})
        self._next_frame[stream_id] = 0

    def enhance(self, image: np.ndarray, *, stream_id: str,
                frame_index: int) -> tuple[np.ndarray, dict]:
        expected = self._next_frame.get(stream_id, 0)
        if frame_index != expected:
            raise EnhancerError(
                f"local stream {stream_id!r} expected frame {expected}, got {frame_index}")
        start = time.perf_counter()
        response = self._request({
            "op": "enhance", "stream_id": stream_id,
            "frame_index": int(frame_index), "image_png_b64": _encode_png(image)})
        result = _decode_png(response["image_png_b64"])
        self._next_frame[stream_id] = expected + 1
        meta = dict(response.get("meta", {}))
        meta.setdefault("round_trip_ms", (time.perf_counter() - start) * 1000.0)
        return result, meta

    def model_info(self) -> dict:
        return dict(self._request({"op": "model_info"}).get("model_info", {}))


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = sock.recv(size - len(chunks))
        if not chunk:
            raise ConnectionError("socket closed before complete response")
        chunks.extend(chunk)
    return bytes(chunks)
