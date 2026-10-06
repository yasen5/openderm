"""Synthetic freehand capture: a textured curved patch seen from arbitrary cameras.

Ground truth is known exactly (surface, intrinsics, every camera pose), which is
what the pose-injection and sim-capture tests need and what a real video lacks.
Conventions match processing: millimetres, camera-to-world rotations, OpenCV
camera axes (+x right, +y down, +z forward), world +z toward the cameras.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from processing.registration_surface import Surface, ray_surface_intersect

FloatArray = NDArray[np.float64]


@dataclass
class Camera:
    R: FloatArray  # camera-to-world (3,3)
    C: FloatArray  # centre, mm (3,)


@dataclass
class Scene:
    surface: Surface
    texture: NDArray[np.uint8]  # (H, W, 3), BGR
    texture_extent_mm: tuple[float, float, float, float]  # xmin, xmax, ymin, ymax
    fx: float
    k1: float
    size: tuple[int, int]  # (width, height)


def look_at(center: FloatArray, target: FloatArray, roll_deg: float = 0.0) -> FloatArray:
    """Camera-to-world rotation looking from ``center`` at ``target``."""
    forward = target - center
    forward = forward / np.linalg.norm(forward)
    up_hint = np.array([0.0, 1.0, 0.0]) if abs(forward[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    down = -(up_hint - (up_hint @ forward) * forward)
    down = down / np.linalg.norm(down)
    right = np.cross(down, forward)
    rotation = np.stack([right, down, forward], axis=1)
    roll = np.radians(roll_deg)
    spin = np.array(
        [[np.cos(roll), -np.sin(roll), 0.0], [np.sin(roll), np.cos(roll), 0.0], [0.0, 0.0, 1.0]]
    )
    return rotation @ spin


def make_scene(seed: int = 0, fx: float = 900.0, k1: float = 0.0, size: tuple[int, int] = (640, 480)) -> Scene:
    """A ~90x70 mm gently curved patch with multi-scale skin-like texture."""
    rng = np.random.default_rng(seed)
    xs = np.arange(-60.0, 60.0 + 1e-6, 2.0)
    ys = np.arange(-50.0, 50.0 + 1e-6, 2.0)
    grid_x, grid_y = np.meshgrid(xs, ys)
    heights = 6.0 * np.cos(grid_x / 45.0) + 4.0 * np.sin(grid_y / 38.0) + 0.02 * grid_x
    surface = Surface(xs, ys, heights)

    ppmm = 10.0  # close to the ~8.6 px/mm the renderer samples at, so no aliasing
    width, height = int((xs[-1] - xs[0]) * ppmm), int((ys[-1] - ys[0]) * ppmm)
    texture = np.zeros((height, width), np.float32)
    for sigma, weight in ((1.6, 0.7), (4.0, 1.0), (10.0, 1.2)):
        octave = cv2.GaussianBlur(rng.standard_normal((height, width)).astype(np.float32), (0, 0), sigma)
        texture += weight * octave / octave.std()
    texture = (127.0 + 45.0 * texture / texture.std()).clip(0, 255)
    tinted = np.stack([texture * 0.80, texture * 0.90, texture], axis=-1).astype(np.uint8)
    return Scene(surface, tinted, (xs[0], xs[-1], ys[0], ys[-1]), fx, k1, size)


def render_view(scene: Scene, camera: Camera) -> NDArray[np.uint8]:
    """Render the textured surface from ``camera`` (pinhole + k1, no occlusion)."""
    width, height = scene.size
    pixel_x, pixel_y = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64))
    distorted = np.stack([(pixel_x - width / 2) / scene.fx, (pixel_y - height / 2) / scene.fx], -1).reshape(-1, 2)
    normalized = distorted.copy()
    for _ in range(5):  # invert f*x*(1 + k1 r^2) by fixed-point iteration
        normalized = distorted / (1.0 + scene.k1 * (normalized**2).sum(1, keepdims=True))
    rays_cam = np.concatenate([normalized, np.ones((len(normalized), 1))], 1)
    rays_world = rays_cam @ camera.R.T
    rays_world /= np.linalg.norm(rays_world, axis=1, keepdims=True)
    distance_prior = float(np.linalg.norm(camera.C))
    hits = ray_surface_intersect(camera.C, rays_world, scene.surface, max(distance_prior, 1.0))
    xmin, xmax, ymin, ymax = scene.texture_extent_mm
    texture_height, texture_width = scene.texture.shape[:2]
    map_x = ((hits[:, 0] - xmin) / (xmax - xmin) * (texture_width - 1)).astype(np.float32)
    map_y = ((hits[:, 1] - ymin) / (ymax - ymin) * (texture_height - 1)).astype(np.float32)
    image = cv2.remap(
        scene.texture,
        map_x.reshape(height, width),
        map_y.reshape(height, width),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    return np.asarray(cv2.GaussianBlur(image, (0, 0), 0.6), dtype=np.uint8)


def random_freehand_cameras(count: int, seed: int = 1, distance_mm: float = 105.0) -> list[Camera]:
    """Cameras scattered over a one-sided cap with arbitrary tilt, yaw and roll."""
    rng = np.random.default_rng(seed)
    cameras: list[Camera] = []
    for index in range(count):
        # a loose raster so neighbouring frames overlap, plus freehand jitter
        fraction = index / max(count - 1, 1)
        lateral_x = -28.0 + 56.0 * fraction + rng.normal(0, 3.0)
        lateral_y = 14.0 * np.sin(2 * np.pi * fraction * 1.5) + rng.normal(0, 3.0)
        center = np.array([lateral_x, lateral_y, distance_mm * rng.uniform(0.88, 1.12)])
        aim = np.array([lateral_x * 0.25 + rng.normal(0, 6.0), lateral_y * 0.25 + rng.normal(0, 6.0), 0.0])
        cameras.append(Camera(look_at(center, aim, roll_deg=float(rng.uniform(-25.0, 25.0))), center))
    return cameras
