"""Unix-socket service for NVIDIA Harmonizer with per-stream temporal state.

Run inside the Harmonizer environment/container. The SimAny process communicates
through a length-prefixed JSON protocol and never imports Cosmos/TensorRT.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import socketserver
import struct
import threading
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


def _encode(image: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB").save(buffer, "PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _decode(value: str) -> np.ndarray:
    with Image.open(io.BytesIO(base64.b64decode(value))) as image:
        return np.asarray(image.convert("RGB"))


def _recv_exact(stream, size: int) -> bytes:
    data = bytearray()
    while len(data) < size:
        chunk = stream.read(size - len(data))
        if not chunk:
            raise ConnectionError("request ended early")
        data.extend(chunk)
    return bytes(data)


class IdentityBackend:
    name = "identity"

    def reset_stream(self, stream_id: str) -> None:
        return None

    def enhance(self, image: np.ndarray, stream_id: str) -> np.ndarray:
        return image.copy()

    def info(self) -> dict:
        return {"backend": self.name, "scientific_result": False}


class NvidiaPix2PixBackend:
    """Thin adapter around NVIDIA's released Pix2Pix_Turbo class."""

    name = "nvidia_pix2pix_harmonizer"
    RESOLUTION_MAP = {1024: (1024, 576), 960: (960, 544), 1360: (1360, 768)}

    def __init__(self, source_dir: str, model_path: str, *, temporal: bool = True,
                 resolution: int = 1024, timestep: int = 250,
                 offsets=(-1, -2, -3, -4), device="cuda", dtype="bfloat16"):
        import sys
        sys.path.insert(0, str(Path(source_dir).resolve()))
        import torch
        from einops import rearrange
        from pix2pix_turbo_harmonizer import Pix2Pix_Turbo
        from torchvision import transforms

        self.torch = torch
        self.rearrange = rearrange
        self.transforms = transforms
        self.device = torch.device(device)
        self.dtype = getattr(torch, dtype)
        self.temporal = bool(temporal)
        self.size = self.RESOLUTION_MAP[int(resolution)]
        self.offsets = tuple(int(v) for v in offsets)
        if any(v >= 0 for v in self.offsets):
            raise ValueError("temporal offsets must refer to past frames")
        self.min_history = -min(self.offsets)
        self.history: dict[str, list[np.ndarray]] = defaultdict(list)
        self.lock = threading.Lock()
        self.model = Pix2Pix_Turbo(
            pretrained_path=model_path, timestep=int(timestep),
            train_full_unet=True, freeze_vae=False,
            vae_skip_connection=False, use_sched=False,
            device=self.device, dtype=self.dtype).to(
                device=self.device, dtype=self.dtype)
        self.model.set_eval()
        self.model_path = str(model_path)
        self.resolution = int(resolution)
        self.timestep = int(timestep)

    def reset_stream(self, stream_id: str) -> None:
        with self.lock:
            self.history.pop(stream_id, None)

    def _tensor(self, image: np.ndarray):
        pil = Image.fromarray(image, mode="RGB").resize(self.size, Image.BILINEAR)
        tensor = self.transforms.ToTensor()(pil)
        return self.transforms.Normalize([0.5], [0.5])(tensor).unsqueeze(0).to(
            device=self.device, dtype=self.dtype)

    def enhance(self, image: np.ndarray, stream_id: str) -> np.ndarray:
        torch = self.torch
        current = self._tensor(image)
        with self.lock:
            history = list(self.history[stream_id])
        if self.temporal and len(history) >= self.min_history:
            refs = [self._tensor(history[offset]) for offset in self.offsets]
            stacked = torch.stack([current] + refs, dim=1)
            stacked = self.rearrange(stacked, "b v c h w -> b c v h w")
        else:
            stacked = self.rearrange(current.unsqueeze(1), "b v c h w -> b c v h w")
        with torch.no_grad():
            output = self.model(stacked).float()
        output = self.rearrange(output, "b c v h w -> b v c h w")[0, 0]
        output = torch.clamp(output * 0.5 + 0.5, 0.0, 1.0)
        pil = self.transforms.ToPILImage()(output.cpu()).resize(
            (image.shape[1], image.shape[0]), Image.BILINEAR)
        result = np.asarray(pil.convert("RGB"))
        with self.lock:
            self.history[stream_id].append(result)
        return result

    def info(self) -> dict:
        return {"backend": self.name, "model_path": self.model_path,
                "temporal": self.temporal, "resolution": self.resolution,
                "timestep": self.timestep, "offsets": list(self.offsets)}


class Handler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        try:
            request_size = struct.unpack("!Q", _recv_exact(self.rfile, 8))[0]
            request = json.loads(_recv_exact(self.rfile, request_size))
            response = self.server.dispatch(request)
        except Exception as exc:
            response = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        body = json.dumps(response, separators=(",", ":")).encode("utf-8")
        self.wfile.write(struct.pack("!Q", len(body)) + body)


class Server(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, socket_path: str, backend):
        self.backend = backend
        self.next_frame: dict[str, int] = {}
        self.lock = threading.Lock()
        super().__init__(socket_path, Handler)

    def dispatch(self, request: dict) -> dict:
        op = request.get("op")
        if op == "model_info":
            return {"ok": True, "model_info": self.backend.info()}
        if op == "reset_stream":
            stream_id = str(request["stream_id"])
            self.backend.reset_stream(stream_id)
            with self.lock:
                self.next_frame[stream_id] = 0
            return {"ok": True}
        if op != "enhance":
            raise ValueError(f"unknown operation {op!r}")
        stream_id = str(request["stream_id"])
        frame_index = int(request["frame_index"])
        with self.lock:
            expected = self.next_frame.get(stream_id, 0)
            if frame_index != expected:
                raise ValueError(
                    f"stream {stream_id!r} expected frame {expected}, got {frame_index}")
        image = _decode(request["image_png_b64"])
        start = time.perf_counter()
        output = self.backend.enhance(image, stream_id)
        latency_ms = (time.perf_counter() - start) * 1000.0
        with self.lock:
            self.next_frame[stream_id] = expected + 1
        return {"ok": True, "image_png_b64": _encode(output),
                "meta": {"latency_ms": latency_ms, **self.backend.info()}}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", required=True)
    parser.add_argument("--backend", choices=("identity", "nvidia"), default="identity")
    parser.add_argument("--source-dir")
    parser.add_argument("--model-path")
    parser.add_argument("--nontemporal", action="store_true")
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--timestep", type=int, default=250)
    args = parser.parse_args(argv)
    socket_path = Path(args.socket)
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    if socket_path.exists():
        socket_path.unlink()
    if args.backend == "identity":
        backend = IdentityBackend()
    else:
        if not args.source_dir or not args.model_path:
            parser.error("--source-dir and --model-path are required for NVIDIA backend")
        backend = NvidiaPix2PixBackend(
            args.source_dir, args.model_path, temporal=not args.nontemporal,
            resolution=args.resolution, timestep=args.timestep)
    server = Server(str(socket_path), backend)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        if socket_path.exists():
            socket_path.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
