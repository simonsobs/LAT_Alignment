"""
Functions for coordinate transforms.

There are 6 relevant coordinate systems here, belonging to two sets of three.
Each set is a global, a primary, and a secondary coordinate system;
where primary and secondary are internal to those mirrors.
The two sets of coordinates are the optical coordinates and the coordinates used
by vertex. We denote these six coordinate systems as follows:

    - opt_global
    - opt_primary
    - opt_secondary
    - va_global
    - va_primary
    - va_secondary
"""

import logging
from functools import cache

import numpy as np
from megham.transform import apply_transform, get_affine
from numpy.typing import NDArray

logger = logging.getLogger("lat_alignment")

opt_sm1 = np.array((0, 0, 3600), np.float64)  # mm
opt_sm2 = np.array((0, -4800, 0), np.float64)  # mm
opt_am1 = -np.arctan(0.5)
opt_am2 = np.arctan(1.0 / 3.0) - np.pi / 2
va_am2 = np.arctan(3.0)
va_sm1 = np.array((-120, 0, -3600), np.float64)  # mm
va_sm2 = np.array((-4920, 0, 0), np.float64)  # mm
va_am1 = np.arctan(0.5)
va_am2 = np.arctan(3.0)
vg2og_shift = (-120, 0, 0)


@cache
def _opt_rot_mat(angle: float) -> NDArray[np.float64]:
    rot_mat = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(angle), np.sin(angle)],
            [0.0, -np.sin(angle), np.cos(angle)],
        ],
        dtype=np.float64,
    )
    return rot_mat


def _opt_global_to_mirror(
    coords: NDArray[np.float64], angle: float, shift: NDArray[np.float64]
):
    rot = _opt_rot_mat(angle)
    return (coords - shift) @ rot.T


def _opt_mirror_to_global(
    coords: NDArray[np.float64], angle: float, shift: NDArray[np.float64]
):
    rot = _opt_rot_mat(angle)
    return coords @ rot + shift


def _opt_global_to_opt_primary(coords: NDArray[np.float64]) -> NDArray[np.float64]:
    return _opt_global_to_mirror(coords, opt_am1, opt_sm1)


def _opt_global_to_opt_secondary(coords: NDArray[np.float64]) -> NDArray[np.float64]:
    return _opt_global_to_mirror(coords, opt_am2, opt_sm2)


def _opt_primary_to_opt_global(coords: NDArray[np.float64]) -> NDArray[np.float64]:
    return _opt_mirror_to_global(coords, opt_am1, opt_sm1)


def _opt_secondary_to_opt_global(coords: NDArray[np.float64]) -> NDArray[np.float64]:
    return _opt_mirror_to_global(coords, opt_am2, opt_sm2)


def _opt_primary_to_opt_secondary(coords: NDArray[np.float64]) -> NDArray[np.float64]:
    return _opt_global_to_mirror(
        _opt_mirror_to_global(coords, opt_am1, opt_sm1), opt_am2, opt_sm2
    )


def _opt_secondary_to_opt_primary(coords: NDArray[np.float64]) -> NDArray[np.float64]:
    return _opt_global_to_mirror(
        _opt_mirror_to_global(coords, opt_am2, opt_sm2), opt_am1, opt_sm1
    )


@cache
def _va_rot_mat(angle: float, sign: int) -> NDArray[np.float64]:
    rot_mat = np.array(
        [
            [sign * np.cos(angle), 0, sign * np.sin(angle)],
            [0, 1, 0],
            [sign * -1 * np.sin(angle), 0, sign * np.cos(angle)],
        ],
        dtype=np.float64,
    )
    return rot_mat


def _va_global_to_mirror(
    coords: NDArray[np.float64], angle: float, sign: int, shift: NDArray[np.float64]
):
    rot = _va_rot_mat(angle, sign)
    return (coords - shift) @ rot.T


def _va_mirror_to_global(
    coords: NDArray[np.float64], angle: float, sign: int, shift: NDArray[np.float64]
):
    rot = _va_rot_mat(angle, sign)
    return coords @ rot + shift


def _va_global_to_va_primary(coords: NDArray[np.float64]) -> NDArray[np.float64]:
    return _va_global_to_mirror(coords, va_am1, 1, va_sm1)


def _va_global_to_va_secondary(coords: NDArray[np.float64]) -> NDArray[np.float64]:
    return _va_global_to_mirror(coords, va_am2, -1, va_sm2)


def _va_primary_to_va_global(coords: NDArray[np.float64]) -> NDArray[np.float64]:
    return _va_mirror_to_global(coords, va_am1, 1, va_sm1)


def _va_secondary_to_va_global(coords: NDArray[np.float64]) -> NDArray[np.float64]:
    return _va_mirror_to_global(coords, va_am2, -1, va_sm2)


def _va_primary_to_va_secondary(coords: NDArray[np.float64]) -> NDArray[np.float64]:
    return _va_global_to_mirror(
        _va_mirror_to_global(coords, va_am1, 1, va_sm1), va_am2, -1, va_sm2
    )


def _va_secondary_to_va_primary(coords: NDArray[np.float64]) -> NDArray[np.float64]:
    return _va_global_to_mirror(
        _va_mirror_to_global(coords, va_am2, 1, va_sm2), va_am1, 1, va_sm1
    )


def _opt_global_to_va_global(coords: NDArray[np.float64]) -> NDArray[np.float64]:
    coords_transformed = coords[:, [1, 0, 2]].copy()
    coords_transformed[:, 2] *= -1
    coords_transformed = coords_transformed + vg2og_shift
    return coords_transformed


def _opt_global_to_va_primary(coords: NDArray[np.float64]) -> NDArray[np.float64]:
    return _va_global_to_va_primary(_opt_global_to_va_global(coords))


def _opt_global_to_va_secondary(coords: NDArray[np.float64]) -> NDArray[np.float64]:
    return _va_global_to_va_secondary(_opt_global_to_va_global(coords))


def _opt_primary_to_va_global(coords: NDArray[np.float64]) -> NDArray[np.float64]:
    return _opt_global_to_va_global(_opt_primary_to_opt_global(coords))


def _opt_primary_to_va_primary(coords: NDArray[np.float64]) -> NDArray[np.float64]:
    return _opt_global_to_va_primary(_opt_primary_to_opt_global(coords))


def _opt_primary_to_va_secondary(coords: NDArray[np.float64]) -> NDArray[np.float64]:
    return _opt_global_to_va_secondary(_opt_primary_to_opt_global(coords))


def _opt_secondary_to_va_global(coords: NDArray[np.float64]) -> NDArray[np.float64]:
    return _opt_global_to_va_global(_opt_secondary_to_opt_global(coords))


def _opt_secondary_to_va_primary(coords: NDArray[np.float64]) -> NDArray[np.float64]:
    return _opt_global_to_va_primary(_opt_secondary_to_opt_global(coords))


def _opt_secondary_to_va_secondary(coords: NDArray[np.float64]) -> NDArray[np.float64]:
    return _opt_global_to_va_secondary(_opt_secondary_to_opt_global(coords))


def _va_global_to_opt_global(coords: NDArray[np.float64]) -> NDArray[np.float64]:
    coords_transformed = coords - vg2og_shift
    coords_transformed = coords_transformed[:, [1, 0, 2]]
    coords_transformed[:, 2] *= -1
    return coords_transformed


def _va_global_to_opt_primary(coords: NDArray[np.float64]) -> NDArray[np.float64]:
    return _opt_global_to_opt_primary(_va_global_to_opt_global(coords))


def _va_global_to_opt_secondary(coords: NDArray[np.float64]) -> NDArray[np.float64]:
    return _opt_global_to_opt_secondary(_va_global_to_opt_global(coords))


def _va_primary_to_opt_global(coords: NDArray[np.float64]) -> NDArray[np.float64]:
    return _va_global_to_opt_global(_va_primary_to_va_global(coords))


def _va_primary_to_opt_primary(coords: NDArray[np.float64]) -> NDArray[np.float64]:
    return _va_global_to_opt_primary(_va_primary_to_va_global(coords))


def _va_primary_to_opt_secondary(coords: NDArray[np.float64]) -> NDArray[np.float64]:
    return _va_global_to_opt_secondary(_va_primary_to_va_global(coords))


def _va_secondary_to_opt_global(coords: NDArray[np.float64]) -> NDArray[np.float64]:
    return _va_global_to_opt_global(_va_secondary_to_va_global(coords))


def _va_secondary_to_opt_primary(coords: NDArray[np.float64]) -> NDArray[np.float64]:
    return _va_global_to_opt_primary(_va_secondary_to_va_global(coords))


def _va_secondary_to_opt_secondary(coords: NDArray[np.float64]) -> NDArray[np.float64]:
    return _va_global_to_opt_secondary(_va_secondary_to_va_global(coords))


def coord_transform(
    coords: NDArray[np.float64], cfrom: str, cto: str
) -> NDArray[np.float64]:
    """
    Transform between the six defined mirror coordinates:

        - opt_global
        - opt_primary
        - opt_secondary
        - va_global
        - va_primary
        - va_secondary

    Parameters
    ----------
    coords : NDArray[np.float64]
        Coordinates to transform.
        Should be a `(npoint, 3)` array.
    cfrom : str
        The coordinate system that `coords` is currently in.
    cto : str
        The coordinate system to put `coords` into.

    Returns
    -------
    coords_transformed : NDArray[np.float64]
        `coords` transformed into `cto`.
    """
    if cfrom == cto:
        return coords
    match f"{cfrom}-{cto}":
        case "opt_global-opt_primary":
            return _opt_global_to_opt_primary(coords).astype(np.float64)
        case "opt_global-opt_secondary":
            return _opt_global_to_opt_secondary(coords).astype(np.float64)
        case "opt_primary-opt_global":
            return _opt_primary_to_opt_global(coords).astype(np.float64)
        case "opt_secondary-opt_global":
            return _opt_secondary_to_opt_global(coords).astype(np.float64)
        case "opt_primary-opt_secondary":
            return _opt_primary_to_opt_secondary(coords).astype(np.float64)
        case "opt_secondary-opt_primary":
            return _opt_secondary_to_opt_primary(coords).astype(np.float64)
        case "va_global-va_primary":
            return _va_global_to_va_primary(coords).astype(np.float64)
        case "va_global-va_secondary":
            return _va_global_to_va_secondary(coords).astype(np.float64)
        case "va_primary-va_global":
            return _va_primary_to_va_global(coords).astype(np.float64)
        case "va_secondary-va_global":
            return _va_secondary_to_va_global(coords).astype(np.float64)
        case "va_primary-va_secondary":
            return _va_primary_to_va_secondary(coords).astype(np.float64)
        case "va_secondary-va_primary":
            return _va_secondary_to_va_primary(coords).astype(np.float64)
        case "opt_global-va_global":
            return _opt_global_to_va_global(coords).astype(np.float64)
        case "opt_global-va_primary":
            return _opt_global_to_va_primary(coords).astype(np.float64)
        case "opt_global-va_secondary":
            return _opt_global_to_va_secondary(coords).astype(np.float64)
        case "opt_primary-va_global":
            return _opt_primary_to_va_global(coords).astype(np.float64)
        case "opt_primary-va_primary":
            return _opt_primary_to_va_primary(coords).astype(np.float64)
        case "opt_primary-va_secondary":
            return _opt_primary_to_va_secondary(coords).astype(np.float64)
        case "opt_secondary-va_global":
            return _opt_secondary_to_va_global(coords).astype(np.float64)
        case "opt_secondary-va_primary":
            return _opt_secondary_to_va_primary(coords).astype(np.float64)
        case "opt_secondary-va_secondary":
            return _opt_secondary_to_va_secondary(coords).astype(np.float64)
        case "va_global-opt_global":
            return _va_global_to_opt_global(coords).astype(np.float64)
        case "va_global-opt_primary":
            return _va_global_to_opt_primary(coords).astype(np.float64)
        case "va_global-opt_secondary":
            return _va_global_to_opt_secondary(coords).astype(np.float64)
        case "va_primary-opt_global":
            return _va_primary_to_opt_global(coords).astype(np.float64)
        case "va_primary-opt_primary":
            return _va_primary_to_opt_primary(coords).astype(np.float64)
        case "va_primary-opt_secondary":
            return _va_primary_to_opt_secondary(coords).astype(np.float64)
        case "va_secondary-opt_global":
            return _va_secondary_to_opt_global(coords).astype(np.float64)
        case "va_secondary-opt_primary":
            return _va_secondary_to_opt_primary(coords).astype(np.float64)
        case "va_secondary-opt_secondary":
            return _va_secondary_to_opt_secondary(coords).astype(np.float64)
        case _:
            raise ValueError("Invalid coordinate system provided!")


def affine_basis_transform(
    aff: NDArray[np.float64],
    sft: NDArray[np.float64],
    cfrom: str,
    cto: str,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Take an affine transform defined in one coordinate system and move it to another.
    The valid coordinate systems are the same as in `coord_transform`.

    Parameters
    ----------
    aff : NDArray[np.float64]
        Affine matrix to tranform.
        Should be a `(3, 3)` array.
    sft : NDArray[np.float64]
        Shift vector to tranform.
        Should be a `(3,)` array.
    cfrom : str
        The coordinate system that `aff` and `sft` is currently in.
    cto : str
        The coordinate system to put `aff` and `sft` into.

    Returns
    -------
    aff_transformed : NDArray[np.float64]
        `aff` transformed into `cto`.
    sft_transformed : NDArray[np.float64]
        `sft` transformed into `cto`.
    """
    # Make a grid of reference points
    line = np.array((-1, 0, 1), np.float64)
    x, y, z = np.meshgrid(line, line, line)
    xyz = np.column_stack((x.ravel(), y.ravel(), z.ravel()))

    # Apply the affine transform
    xyz_transformed = apply_transform(xyz, aff, sft)

    # Move to the new coordinate system
    xyz = coord_transform(xyz, cfrom, cto)
    xyz_transformed = coord_transform(xyz_transformed, cfrom, cto)

    # Get the new affine transform
    aff, sft = get_affine(xyz, xyz_transformed)

    return aff, sft
