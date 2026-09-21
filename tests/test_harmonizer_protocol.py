import threading

import numpy as np

from integrations.harmonizer.server import IdentityBackend, Server
from robo.rendering.harmonizer_client import EnhancerError, SocketHarmonizerClient


def test_socket_protocol_and_frame_order(tmp_path, monkeypatch):
    # AF_UNIX limits the socket name, even when pytest uses a long checkout path.
    monkeypatch.chdir(tmp_path)
    path = "harmonizer.sock"
    server = Server(str(path), IdentityBackend())
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = SocketHarmonizerClient(str(path), timeout_s=2)
        client.reset_stream("episode/camera")
        image = np.full((8, 9, 3), 17, dtype=np.uint8)
        result, _ = client.enhance(
            image, stream_id="episode/camera", frame_index=0)
        assert np.array_equal(result, image)
        try:
            client.enhance(image, stream_id="episode/camera", frame_index=2)
            raise AssertionError("out-of-order frame was accepted")
        except EnhancerError:
            pass
    finally:
        server.shutdown()
        server.server_close()
