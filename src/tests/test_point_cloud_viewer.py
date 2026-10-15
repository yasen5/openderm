"""Export -> HTTP viewer integration, including the resources needed over SSH."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("cv2")

from processing.registration_export import export_colored_point_cloud_ply
from processing.view_point_cloud import create_viewer_server


def test_exported_cloud_is_served_with_local_viewer_and_no_directory_access(tmp_path: Path) -> None:
    cloud = tmp_path / "texture.ply"
    export_colored_point_cloud_ply(
        cloud,
        np.array([[1.0, 2.0, 3.0], [-4.0, 5.0, 6.0]]),
        np.array([[255, 10, 20], [30, 240, 50]], dtype=np.uint8),
    )
    (tmp_path / "private.txt").write_text("not a viewer resource")
    server = create_viewer_server(tmp_path, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(base + "/info.json", timeout=5) as response:
            info = json.load(response)
        assert info["points"] == 2 and info["units"] == "mm"
        assert info["data_offset"] + 2 * 15 == cloud.stat().st_size
        with urlopen(base + "/", timeout=5) as response:
            html = response.read().decode()
            assert response.headers.get_content_type() == "text/html"
        assert '"three":"/three.module.min.js"' in html
        assert 'href="/cloud.ply"' in html
        assert "https://" not in html
        for resource in ("three.module.min.js", "OrbitControls.js"):
            with urlopen(base + "/" + resource, timeout=5) as response:
                assert response.headers.get_content_type() == "text/javascript"
                assert len(response.read()) > 1000
        with urlopen(base + "/cloud.ply", timeout=5) as response:
            assert response.read() == cloud.read_bytes()
        with urlopen(Request(base + "/cloud.ply", method="HEAD"), timeout=5) as response:
            assert int(response.headers["Content-Length"]) == cloud.stat().st_size
            assert response.read() == b""
        for route in ("/private.txt", "/../private.txt", "/%2e%2e/private.txt"):
            with pytest.raises(HTTPError) as error:
                urlopen(base + route, timeout=5)
            assert error.value.code == 404
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    # Reject a truncated binary cloud before starting the HTTP listener.
    cloud.write_bytes(cloud.read_bytes()[:-1])
    with pytest.raises(ValueError, match="truncated"):
        create_viewer_server(cloud, port=0)
