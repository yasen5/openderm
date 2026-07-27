"""Offline self-collision model for the gantry.

Loads cad/robot.urdf, groups each rigid link into a single collision body, and
checks the moving links against the static frame (and each other) with FCL.

Coordinates are the REAL controller values (x,y,z in mm, rx in rad); they are
mapped to CAD/URDF joint values via cad/frame_calibration.json.

Install the ``collision`` dependency group before running this module. It is the
offline oracle used to precompute the collision-free envelope; the runtime
checker only reads the resulting table and does not need FCL.
"""

from __future__ import annotations

import json
import math
import os
from itertools import combinations

import fcl
import trimesh
import yourdfpy

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CAD_DIR = os.path.join(REPO, "cad")
URDF_PATH = os.path.join(CAD_DIR, "robot.urdf")
CALIB_PATH = os.path.join(CAD_DIR, "frame_calibration.json")

# real-controller poses that are KNOWN collision-free (mid-travel, camera ~level).
# Used to seed the allowed-collision matrix (structural overlaps that are by design).
_ACM_SEED_POSES = [
    dict(x=400, y=340, z=200, rx=0.95),
    dict(x=100, y=120, z=100, rx=0.95),
    dict(x=700, y=560, z=350, rx=0.95),
]


class CollisionModel:
    def __init__(self, urdf_path=URDF_PATH, calib_path=CALIB_PATH):
        self.urdf = yourdfpy.URDF.load(urdf_path, load_meshes=False, build_scene_graph=True)
        with open(calib_path) as f:
            self.calib = json.load(f)
        self.base = self.calib["base_link"]

        # combine each link's collision (fallback visual) meshes into one body, in link frame
        self.link_mesh = {}
        for name, link in self.urdf.link_map.items():
            geoms = link.collisions or link.visuals
            parts = []
            for g in geoms:
                mesh = getattr(g.geometry, "mesh", None)
                if mesh is None:
                    continue
                path = os.path.join(CAD_DIR, mesh.filename.replace("package://", ""))
                try:
                    m = trimesh.load(path, force="mesh")
                except Exception:
                    continue
                if mesh.scale is not None:
                    m.apply_scale(mesh.scale)
                if g.origin is not None:
                    m.apply_transform(g.origin)
                parts.append(m)
            if parts:
                self.link_mesh[name] = trimesh.util.concatenate(parts)
        self.links = list(self.link_mesh)

        # collision manager with one object per link
        self.mgr = trimesh.collision.CollisionManager()
        for name, m in self.link_mesh.items():
            self.mgr.add_object(name, m)

        # adjacent (parent/child) link pairs are allowed to touch
        self.allowed = set()
        for j in self.urdf.joint_map.values():
            self.allowed.add(frozenset((j.parent, j.child)))
        # seed allowed matrix from known-safe poses (structural overlaps)
        for p in _ACM_SEED_POSES:
            self.set_pose(**p)
            self.allowed |= self._colliding_pairs()

    # ---- kinematics ----------------------------------------------------
    def real_to_cad(self, x, y, z, rx):
        a = self.calib["axes"]
        return {
            "x": a["x"]["cad_scale"] * (x / 1000.0) + a["x"]["cad_offset"],
            "y": a["y"]["cad_scale"] * (y / 1000.0) + a["y"]["cad_offset"],
            "z": a["z"]["cad_scale"] * (z / 1000.0) + a["z"]["cad_offset"],
            "rx": a["rx"]["cad_scale"] * rx + a["rx"]["cad_offset"],
        }

    def set_pose(self, x, y, z, rx):
        self.urdf.update_cfg(self.real_to_cad(x, y, z, rx))
        for name in self.links:
            self.mgr.set_transform(name, self.urdf.get_transform(name, self.base))

    # ---- collision -----------------------------------------------------
    def _colliding_pairs(self):
        _, names = self.mgr.in_collision_internal(return_names=True)
        return {frozenset(p) for p in names}

    def candidate_pairs(self):
        """Non-allowed (i.e. checkable) link pairs, sorted."""
        return [
            (a, b)
            for a, b in combinations(sorted(self.links), 2)
            if frozenset((a, b)) not in self.allowed
        ]

    def pair_clearances(self, x, y, z, rx):
        """{(link_a, link_b): distance_mm} for every checkable pair at a pose."""
        self.set_pose(x, y, z, rx)
        objs = self.mgr._objs
        out = {}
        for a, b in self.candidate_pairs():
            d = fcl.distance(
                objs[a]["obj"], objs[b]["obj"], fcl.DistanceRequest(), fcl.DistanceResult()
            )
            out[(a, b)] = d * 1000.0  # m -> mm
        return out

    def clearance(self, x, y, z, rx):
        """Minimum distance in millimeters over every non-allowed link pair."""
        pc = self.pair_clearances(x, y, z, rx)
        return min(pc.values()) if pc else math.inf
