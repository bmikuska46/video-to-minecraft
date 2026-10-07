"""Validation and sizing for the preview-to-export contract.

This module deliberately operates only on reconstruction bounds and user input.
It does not inspect or alter point-cloud geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence


AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


class ContractError(ValueError):
    """A preview transform, crop, or scale violates the public contract."""


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def validate_rigid_transform(values: Sequence[float], *, tolerance: float = 1e-5) -> None:
    """Require a finite, row-major, proper rigid 4x4 affine transform."""
    if len(values) != 16 or not all(math.isfinite(value) for value in values):
        raise ContractError("transform must contain 16 finite values")
    if any(abs(actual - expected) > tolerance for actual, expected in zip(values[12:16], (0, 0, 0, 1), strict=True)):
        raise ContractError("transform must be an affine matrix with final row [0, 0, 0, 1]")

    rotation = (
        (values[0], values[1], values[2]),
        (values[4], values[5], values[6]),
        (values[8], values[9], values[10]),
    )
    for row in rotation:
        if abs(_dot(row, row) - 1.0) > tolerance:
            raise ContractError("transform rotation must not contain scale or shear")
    for first, second in ((0, 1), (0, 2), (1, 2)):
        if abs(_dot(rotation[first], rotation[second])) > tolerance:
            raise ContractError("transform rotation must be orthogonal")
    determinant = (
        rotation[0][0] * (rotation[1][1] * rotation[2][2] - rotation[1][2] * rotation[2][1])
        - rotation[0][1] * (rotation[1][0] * rotation[2][2] - rotation[1][2] * rotation[2][0])
        + rotation[0][2] * (rotation[1][0] * rotation[2][1] - rotation[1][1] * rotation[2][0])
    )
    if abs(determinant - 1.0) > tolerance:
        raise ContractError("transform rotation must be proper (determinant +1)")


def transformed_bounds(
    bounds_min: Sequence[float], bounds_max: Sequence[float], transform: Sequence[float]
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Return the AABB of all eight source-bound corners after transformation."""
    validate_rigid_transform(transform)
    corners: list[tuple[float, float, float]] = []
    for x in (bounds_min[0], bounds_max[0]):
        for y in (bounds_min[1], bounds_max[1]):
            for z in (bounds_min[2], bounds_max[2]):
                corners.append(
                    (
                        transform[0] * x + transform[1] * y + transform[2] * z + transform[3],
                        transform[4] * x + transform[5] * y + transform[6] * z + transform[7],
                        transform[8] * x + transform[9] * y + transform[10] * z + transform[11],
                    )
                )
    return (
        tuple(min(point[axis] for point in corners) for axis in range(3)),
        tuple(max(point[axis] for point in corners) for axis in range(3)),
    )


def validate_crop_within_bounds(
    crop_min: Sequence[float],
    crop_max: Sequence[float],
    available_min: Sequence[float],
    available_max: Sequence[float],
    *,
    tolerance: float = 1e-6,
) -> None:
    for axis in range(3):
        if crop_min[axis] < available_min[axis] - tolerance or crop_max[axis] > available_max[axis] + tolerance:
            raise ContractError("crop must be contained in the transformed reconstruction bounds")


@dataclass(frozen=True)
class BlockEstimate:
    voxel_size: float
    voxel_dimensions: tuple[int, int, int]
    block_count_upper_bound: int


def estimate_blocks(
    crop_min: Sequence[float], crop_max: Sequence[float], *, axis: str, blocks: int
) -> BlockEstimate:
    """Compute voxel size and a conservative occupied-cell upper bound.

    Point-cloud support decides the true occupied count during voxelization.  The
    full crop-grid volume is therefore the only safe synchronous preflight bound.
    """
    extent = tuple(crop_max[index] - crop_min[index] for index in range(3))
    voxel_size = extent[AXIS_INDEX[axis]] / blocks
    dimensions = tuple(max(1, math.ceil(value / voxel_size - 1e-12)) for value in extent)
    return BlockEstimate(
        voxel_size=voxel_size,
        voxel_dimensions=dimensions,
        block_count_upper_bound=math.prod(dimensions),
    )
