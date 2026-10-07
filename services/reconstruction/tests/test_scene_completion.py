from __future__ import annotations

import importlib.util
import json
import math
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


SERVICE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE))


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, SERVICE / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


depth_completion = load("depth_completion")
point_cloud = load("point_cloud")
MODULE = load("scene_completion")

try:
    import torch  # noqa: F401
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# A 4 x 2.5 x 3 room with a bed, a window region on the +x wall that no frame
# ever measures, and a ghost wall in the middle of the room that three frames
# report but every other frame sees through.
ROOM = (np.array([0.0, 0.0, 0.0]), np.array([4.0, 2.5, 3.0]))
BED = (np.array([2.4, 0.0, 0.5]), np.array([3.4, 0.5, 2.0]))
WINDOW = (np.array([3.99, 0.8, 1.0]), np.array([4.01, 1.6, 2.0]))
GHOST = (np.array([1.2, 0.6, 1.0]), np.array([1.25, 1.8, 2.0]))
GHOST_FRAMES = {3, 4, 5}
WIDTH, HEIGHT, FOCAL = 96, 72, 44.0
COLORS = {"floor": (150, 100, 60), "ceiling": (235, 235, 230), "wall": (200, 200, 190), "bed": (180, 40, 40)}


def look_rotation(forward: np.ndarray) -> np.ndarray:
    """World-to-camera rotation for COLMAP axes (x right, y down, z forward), no roll."""
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, [0.0, 1.0, 0.0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    return np.stack([right, down, forward])


def quaternion(rotation: np.ndarray) -> tuple[float, float, float, float]:
    w = math.sqrt(max(0.0, 1 + rotation[0, 0] + rotation[1, 1] + rotation[2, 2])) / 2
    x = math.copysign(math.sqrt(max(0.0, 1 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2])) / 2,
                      rotation[2, 1] - rotation[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1 - rotation[0, 0] + rotation[1, 1] - rotation[2, 2])) / 2,
                      rotation[0, 2] - rotation[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1 - rotation[0, 0] - rotation[1, 1] + rotation[2, 2])) / 2,
                      rotation[1, 0] - rotation[0, 1])
    return w, x, y, z


def box_entry(origin: np.ndarray, directions: np.ndarray, box: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """Ray distance to where each ray enters an axis-aligned box (inf if it misses)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = (box[0] - origin) / directions
        t2 = (box[1] - origin) / directions
    near = np.nanmax(np.minimum(t1, t2), axis=1)
    far = np.nanmin(np.maximum(t1, t2), axis=1)
    return np.where((far >= near) & (near > 1e-6), near, np.inf)


def render(center: np.ndarray, rotation: np.ndarray, ghost: bool) -> tuple[np.ndarray, np.ndarray]:
    v, u = np.mgrid[0:HEIGHT, 0:WIDTH].astype(float)
    camera = np.stack([(u - WIDTH / 2) / FOCAL, (v - HEIGHT / 2) / FOCAL, np.ones_like(u)], -1).reshape(-1, 3)
    directions = camera @ rotation
    # Rays leave the room box from inside: exit distance.
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = (ROOM[0] - center) / directions
        t2 = (ROOM[1] - center) / directions
    distance = np.nanmin(np.where(np.maximum(t1, t2) > 0, np.maximum(t1, t2), np.inf), axis=1)
    hit = center + directions * distance[:, None]
    color = np.empty((len(distance), 3))
    color[:] = COLORS["wall"]
    color[hit[:, 1] < 1e-3] = COLORS["floor"]
    color[hit[:, 1] > ROOM[1][1] - 1e-3] = COLORS["ceiling"]
    unmeasured = ((hit >= WINDOW[0]) & (hit <= WINDOW[1])).all(axis=1)
    bed = box_entry(center, directions, BED)
    on_bed = bed < distance
    distance = np.where(on_bed, bed, distance)
    color[on_bed] = COLORS["bed"]
    unmeasured &= ~on_bed
    if ghost:
        phantom = box_entry(center, directions, GHOST)
        distance = np.where(phantom < distance, phantom, distance)
    depth = (distance * camera[:, 2]).reshape(HEIGHT, WIDTH)
    depth[unmeasured.reshape(HEIGHT, WIDTH)] = 0.0
    return depth.astype(np.float32), color.reshape(HEIGHT, WIDTH, 3).astype(np.uint8)


def write_workspace(dense: Path) -> list[np.ndarray]:
    """COLMAP dense workspace (binary model, images, geometric depth maps)."""
    for folder in ("sparse", "images", "stereo/depth_maps", "stereo/normal_maps"):
        (dense / folder).mkdir(parents=True)
    poses = []
    for index in range(36):
        # A walk around the room looking at the walls and down at the floor, then
        # a few frames up at the ceiling.
        angle = 2 * math.pi * index / 24
        center = np.array([2.0 + 0.9 * math.cos(angle), 1.4, 1.5 + 0.6 * math.sin(angle)])
        heading = angle + (math.pi if index % 2 else 0.0) + 0.4 * (index % 3 - 1)
        pitch = (-0.75, -0.35)[index % 2] if index < 28 else 0.55
        forward = np.array([math.cos(heading) * math.cos(pitch), math.sin(pitch),
                            math.sin(heading) * math.cos(pitch)])
        if index in GHOST_FRAMES:
            forward = (GHOST[0] + GHOST[1]) / 2 - center
        poses.append((f"frame-{index + 1:04d}.png", center, look_rotation(forward), index in GHOST_FRAMES))
    with (dense / "sparse" / "cameras.bin").open("wb") as cameras:
        cameras.write(struct.pack("<Q", 1))
        cameras.write(struct.pack("<iiQQ", 1, 1, WIDTH, HEIGHT))
        cameras.write(struct.pack("<4d", FOCAL, FOCAL, WIDTH / 2, HEIGHT / 2))
    with (dense / "sparse" / "images.bin").open("wb") as images:
        images.write(struct.pack("<Q", len(poses)))
        for image_id, (name, center, rotation, _) in enumerate(poses, start=1):
            translation = -rotation @ center
            images.write(struct.pack("<i7di", image_id, *quaternion(rotation), *translation, 1))
            images.write(name.encode() + b"\0")
            images.write(struct.pack("<Q", 0))
    (dense / "sparse" / "points3D.bin").write_bytes(struct.pack("<Q", 0))
    centers = []
    for name, center, rotation, ghost in poses:
        depth, color = render(center, rotation, ghost)
        Image.fromarray(color).save(dense / "images" / name)
        depth_completion.write_colmap_array(dense / "stereo" / "depth_maps" / f"{name}.geometric.bin", depth)
        centers.append(center)
    return centers


def write_measured_cloud(path: Path) -> None:
    """Sparse room-surface samples standing in for the filtered COLMAP fusion."""
    rng = np.random.default_rng(0)
    samples = []
    for axis in range(3):
        for side in (0, 1):
            points = rng.uniform(ROOM[0], ROOM[1], (300, 3))
            points[:, axis] = ROOM[side][axis]
            normal = np.zeros(3)
            normal[axis] = 1.0 if side == 0 else -1.0
            samples.append(np.hstack([points, np.repeat(normal[None], len(points), 0)]))
    data = np.vstack(samples)
    dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
                      ("red", "u1"), ("green", "u1"), ("blue", "u1"), ("observed", "u1")])
    cloud = np.zeros(len(data), dtype)
    for index, name in enumerate(("x", "y", "z", "nx", "ny", "nz")):
        cloud[name] = data[:, index]
    cloud["red"] = cloud["green"] = cloud["blue"] = 200
    point_cloud.write_ply(path, cloud)


def distance_to_room_boundary(xyz: np.ndarray) -> np.ndarray:
    return np.minimum(np.abs(xyz - ROOM[0]), np.abs(xyz - ROOM[1])).min(axis=1)


class HelperTests(unittest.TestCase):
    def test_wall_alignment_recovers_the_room_yaw(self):
        yaw = math.radians(23.0)
        angles = np.repeat(np.arange(4) * math.pi / 2, 50) - yaw
        normals = np.stack([np.cos(angles), np.zeros_like(angles), np.sin(angles)], 1)
        found = MODULE.wall_alignment_yaw(normals)
        self.assertAlmostEqual(math.degrees(found), 23.0, delta=1.5)
        turned = normals @ MODULE.yaw_rotation(found).T
        # Every wall normal ends up on the X or Z axis.
        self.assertLess(np.abs(turned[:, [0, 2]]).min(axis=1).max(), 0.03)

    def test_harmonic_fill_continues_a_sloped_ceiling(self):
        x, z = np.mgrid[0:30, 0:20]
        plane = 40 + 0.5 * x - 0.25 * z
        # The unseen middle is enclosed by seen cells, as a ceiling seen all
        # around the room's edge.
        known = np.ones_like(plane, bool)
        known[5:-5, 5:-5] = False
        filled = MODULE.harmonic_fill(np.where(known, plane, 0.0), known, np.ones_like(known))
        np.testing.assert_allclose(filled, plane, atol=1e-3)

    def test_frustum_box_holds_every_voxel_the_frame_sees(self):
        rng = np.random.default_rng(3)
        origin, voxel, shape = np.array([-2.0, -1.5, -2.5]), 0.1, (40, 30, 50)
        index = np.argwhere(np.ones(shape, bool))
        centers = origin + (index + 0.5) * voxel
        for _ in range(5):
            angle = rng.uniform(0, 2 * math.pi)
            rotation = np.array([[math.cos(angle), 0, -math.sin(angle)], [0, 1, 0], [math.sin(angle), 0, math.cos(angle)]])
            center = rng.uniform(-1.0, 1.0, 3)
            frame = {"depth": np.ones((30, 40), np.float32), "K": (30.0, 30.0, 20.0, 15.0),
                     "R": rotation, "t": -rotation @ center}
            far = rng.uniform(0.5, 6.0)
            camera = centers @ rotation.T + frame["t"]
            z = camera[:, 2]
            safe = np.where(z > 1e-6, z, 1.0)
            u = np.round(30.0 * camera[:, 0] / safe + 20.0)
            v = np.round(30.0 * camera[:, 1] / safe + 15.0)
            seen = (z > 1e-6) & (z <= far) & (u >= 0) & (u < 40) & (v >= 0) & (v < 30)
            low, high = MODULE.frustum_box(frame, origin, voxel, shape, far)
            inside = ((index >= low) & (index < high)).all(axis=1)
            self.assertTrue(inside[seen].all())
            self.assertLess(inside.sum(), len(index))


@unittest.skipUnless(HAS_TORCH, "scene completion needs torch")
class SyntheticRoomTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        root = Path(cls.temporary.name)
        cls.root = root
        write_workspace(root / "dense")
        write_measured_cloud(root / "measured.ply")
        (root / "filter-metrics.json").write_text(json.dumps({"rotation": np.eye(3).tolist()}))
        cls.metrics = MODULE.complete_scene(
            root / "measured.ply", root / "dense", root / "filter-metrics.json",
            root / "canonical.ply", root / "preview.ply", resolution=64, device="cpu",
        )
        (root / "scene-completion.json").write_text(json.dumps(cls.metrics))
        cloud = point_cloud.read_ply(root / "canonical.ply")
        # The synthetic room is already wall aligned; undo the (zero) yaw anyway.
        rotation = np.asarray(cls.metrics["rotation"])
        cls.xyz = np.column_stack([cloud[name] for name in "xyz"]).astype(float) @ rotation
        cls.synthesized = cloud["synthesized"] > 0
        cls.colors = np.column_stack([cloud[name] for name in ("red", "green", "blue")]).astype(float)
        cls.voxel = cls.metrics["voxelSize"]

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_room_is_wall_aligned(self):
        self.assertLess(abs(self.metrics["yawDegrees"]), 1.0)

    def test_shell_lies_on_the_room_boundary_and_nothing_is_added_inside(self):
        self.assertTrue(self.synthesized.any())
        boundary = distance_to_room_boundary(self.xyz[self.synthesized])
        self.assertLess(np.percentile(boundary, 99), 2.5 * self.voxel)
        self.assertLess(boundary.max(), 4 * self.voxel)

    def test_never_measured_window_is_closed_with_wall_color(self):
        window = ((self.xyz > WINDOW[0] - [3 * self.voxel, -0.1, -0.1])
                  & (self.xyz < WINDOW[1] + [3 * self.voxel, -0.1, -0.1])).all(axis=1)
        self.assertGreater(int((window & self.synthesized).sum()), 20)
        self.assertFalse((window & ~self.synthesized).any(), "window region has no measured surface")
        mean = self.colors[window].mean(axis=0)
        np.testing.assert_allclose(mean, COLORS["wall"], atol=25)

    def test_bed_is_kept_as_an_object_not_walled_in(self):
        top = ((np.abs(self.xyz[:, 1] - BED[1][1]) < 1.5 * self.voxel)
               & (self.xyz[:, 0] > BED[0][0] + 0.1) & (self.xyz[:, 0] < BED[1][0] - 0.1)
               & (self.xyz[:, 2] > BED[0][2] + 0.1) & (self.xyz[:, 2] < BED[1][2] - 0.1))
        self.assertGreater(int((top & ~self.synthesized).sum()), 50)
        self.assertFalse((top & self.synthesized).any())
        self.assertGreater(np.median(self.colors[top][:, 0]) - np.median(self.colors[top][:, 1]), 80)

    def test_ghost_wall_seen_by_three_frames_is_carved_away(self):
        ghost = ((self.xyz > GHOST[0] - 2 * self.voxel) & (self.xyz < GHOST[1] + 2 * self.voxel)).all(axis=1)
        self.assertEqual(int(ghost.sum()), 0)


if __name__ == "__main__":
    unittest.main()
