"""Serve a textured surface viewer or RGB point cloud on localhost through SSH.

    python -m processing.view_point_cloud captures/cleaner-texture3d --port 8765

On your other machine:
    ssh -N -L 8765:127.0.0.1:8765 USER@SERVER
Then open http://localhost:8765. Ctrl-C stops the viewer server.
"""

from __future__ import annotations

import argparse
import json
import shutil
import socket
import sysconfig
from collections.abc import Sequence
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from socketserver import BaseServer
from typing import TypedDict
from urllib.parse import urlsplit


class CloudInfo(TypedDict):
    name: str
    points: int
    data_offset: int
    units: str


def inspect_point_cloud(path: Path) -> CloudInfo:
    """Validate the fixed binary XYZ/RGB layout written by our PLY exporter."""
    expected = [
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
    ]
    with path.open("rb") as stream:
        header = bytearray()
        while len(header) < 65536:
            line = stream.readline(1024)
            if not line:
                break
            header.extend(line)
            if line.strip() == b"end_header":
                break
        else:
            raise ValueError("PLY header is too large")
    try:
        lines = header.decode("ascii").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError("invalid PLY header") from error
    if len(lines) < 4 or lines[:2] != ["ply", "format binary_little_endian 1.0"]:
        raise ValueError("expected a binary little-endian PLY from openderm-texture-3d")
    vertex_lines = [line for line in lines if line.startswith("element vertex ")]
    if len(vertex_lines) != 1 or not lines or lines[-1] != "end_header":
        raise ValueError("missing vertex count or end_header")
    try:
        count = int(vertex_lines[0].split()[-1])
    except ValueError as error:
        raise ValueError("invalid PLY vertex count") from error
    properties = [line for line in lines if line.startswith("property ")]
    elements = [line for line in lines if line.startswith("element ")]
    if properties != expected or elements != vertex_lines:
        raise ValueError("expected XYZ float coordinates and RGB byte colors, with no faces")
    if count <= 0 or path.stat().st_size != len(header) + count * 15:
        raise ValueError("PLY is empty, truncated, or has an unexpected record size")
    return CloudInfo(name=path.parent.name, points=count, data_offset=len(header), units="mm")


def threejs_directory() -> Path:
    """Find the existing vendored Three.js files in a checkout or installation."""
    repository = Path(__file__).resolve().parents[2] / "third_party" / "threejs"
    installed = Path(sysconfig.get_path("data")) / "share" / "openderm" / "third_party" / "threejs"
    for directory in (repository, installed):
        if all(
            (directory / name).is_file() for name in ("three.module.min.js", "OrbitControls.js")
        ):
            return directory
    raise ValueError(
        "vendored Three.js files are missing; reinstall openderm or use a full checkout"
    )


class PointCloudHandler(BaseHTTPRequestHandler):
    """Serve only the viewer, its two library files, and the selected cloud."""

    def __init__(
        self,
        request: socket.socket,
        client_address: tuple[str, int],
        server: BaseServer,
        *,
        files: dict[str, Path],
        info: CloudInfo | None,
    ) -> None:
        self.files = files
        self.info = info
        super().__init__(request, client_address, server)

    def do_GET(self) -> None:
        self._respond(send_body=True)

    def do_HEAD(self) -> None:
        self._respond(send_body=False)

    def _respond(self, *, send_body: bool) -> None:
        route = urlsplit(self.path).path
        if route == "/info.json" and self.info is not None:
            body = json.dumps(self.info).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            if send_body:
                self.wfile.write(body)
            return
        path = self.files.get(route)
        if path is None:
            self.send_error(404)
            return
        content_type = (
            "text/html; charset=utf-8"
            if route == "/"
            else "text/javascript; charset=utf-8"
            if route.endswith(".js")
            else "application/octet-stream"
        )
        try:
            # Open before sending headers so an unavailable file gets a proper error.
            with path.open("rb") as stream:
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(path.stat().st_size))
                self.send_header("Cache-Control", "no-cache")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.end_headers()
                if send_body:
                    shutil.copyfileobj(stream, self.wfile)
        except (BrokenPipeError, ConnectionResetError):
            pass  # browser closed a large download
        except FileNotFoundError:
            self.send_error(404)


def create_viewer_server(
    source: Path, host: str = "127.0.0.1", port: int = 8765
) -> ThreadingHTTPServer:
    viewer = source / "viewer.html" if source.is_dir() else source
    if viewer.name == "viewer.html" and viewer.is_file():
        # Surface exports embed mesh, texture, cameras and vendored Three.js.
        return ThreadingHTTPServer(
            (host, port), partial(PointCloudHandler, files={"/": viewer}, info=None)
        )
    cloud = source / "texture.ply" if source.is_dir() else source
    info = inspect_point_cloud(cloud)
    libraries = threejs_directory()
    files = {
        "/": Path(__file__).with_name("assets") / "point_cloud_viewer.html",
        "/cloud.ply": cloud,
        "/three.module.min.js": libraries / "three.module.min.js",
        "/OrbitControls.js": libraries / "OrbitControls.js",
    }
    return ThreadingHTTPServer((host, port), partial(PointCloudHandler, files=files, info=info))


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("source", type=Path, help="output directory, viewer.html, or texture.ply")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    try:
        server = create_viewer_server(args.source, port=args.port)
    except (OSError, ValueError) as error:
        parser.exit(1, f"openderm-view-3d: {error}\n")
    print(f"Viewing {args.source.resolve()} at http://127.0.0.1:{args.port}", flush=True)
    print(f"SSH tunnel: ssh -N -L {args.port}:127.0.0.1:{args.port} USER@SERVER", flush=True)
    print("Press Ctrl-C to stop.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
