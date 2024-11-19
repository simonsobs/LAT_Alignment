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
from functools import cache, partial

import matplotlib.pyplot as plt
import numpy as np
from megham.transform import apply_transform, get_affine, get_rigid
from megham.utils import make_edm
from numpy.typing import NDArray

logger = logging.getLogger("lat_alignment")

opt_sm1 = np.array((0, 0, 3600), np.float32)  # mm
opt_sm2 = np.array((0, -4800, 0), np.float32)  # mm
opt_am1 = -np.arctan(0.5)
opt_am2 = np.arctan(1.0 / 3.0) - np.pi / 2
va_am2 = np.arctan(3.0)
va_sm1 = np.array((-120, 0, -3600), np.float32)  # mm
va_sm2 = np.array((-4920, 0, 0), np.float32)  # mm
va_am1 = np.arctan(0.5)
va_am2 = np.arctan(3.0)
vg2og_shift = (-120, 0, 0)


@cache
def _opt_rot_mat(angle: float) -> NDArray[np.float32]:
    rot_mat = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(angle), np.sin(angle)],
            [0.0, -np.sin(angle), np.cos(angle)],
        ],
        dtype=np.float32,
    )
    return rot_mat


def _opt_global_to_mirror(
    coords: NDArray[np.float32], angle: float, shift: NDArray[np.float32]
):
    rot = _opt_rot_mat(angle)
    return (coords - shift) @ rot.T


def _opt_mirror_to_global(
    coords: NDArray[np.float32], angle: float, shift: NDArray[np.float32]
):
    rot = _opt_rot_mat(angle)
    return coords @ rot + shift


def _opt_global_to_opt_primary(coords: NDArray[np.float32]) -> NDArray[np.float32]:
    return _opt_global_to_mirror(coords, opt_am1, opt_sm1)


def _opt_global_to_opt_secondary(coords: NDArray[np.float32]) -> NDArray[np.float32]:
    return _opt_global_to_mirror(coords, opt_am2, opt_sm2)


def _opt_primary_to_opt_global(coords: NDArray[np.float32]) -> NDArray[np.float32]:
    return _opt_mirror_to_global(coords, opt_am1, opt_sm1)


def _opt_secondary_to_opt_global(coords: NDArray[np.float32]) -> NDArray[np.float32]:
    return _opt_mirror_to_global(coords, opt_am2, opt_sm2)


def _opt_primary_to_opt_secondary(coords: NDArray[np.float32]) -> NDArray[np.float32]:
    return _opt_global_to_mirror(
        _opt_mirror_to_global(coords, opt_am1, opt_sm1), opt_am2, opt_sm2
    )


def _opt_secondary_to_opt_primary(coords: NDArray[np.float32]) -> NDArray[np.float32]:
    return _opt_global_to_mirror(
        _opt_mirror_to_global(coords, opt_am2, opt_sm2), opt_am1, opt_sm1
    )


@cache
def _va_rot_mat(angle: float, sign: int) -> NDArray[np.float32]:
    rot_mat = np.array(
        [
            [sign * np.cos(angle), 0, sign * np.sin(angle)],
            [0, 1, 0],
            [sign * -1 * np.sin(angle), 0, sign * np.cos(angle)],
        ],
        dtype=np.float32,
    )
    return rot_mat


def _va_global_to_mirror(
    coords: NDArray[np.float32], angle: float, sign: int, shift: NDArray[np.float32]
):
    rot = _va_rot_mat(angle, sign)
    return (coords - shift) @ rot.T


def _va_mirror_to_global(
    coords: NDArray[np.float32], angle: float, sign: int, shift: NDArray[np.float32]
):
    rot = _va_rot_mat(angle, sign)
    return coords @ rot + shift


def _va_global_to_va_primary(coords: NDArray[np.float32]) -> NDArray[np.float32]:
    return _va_global_to_mirror(coords, va_am1, 1, va_sm1)


def _va_global_to_va_secondary(coords: NDArray[np.float32]) -> NDArray[np.float32]:
    return _va_global_to_mirror(coords, va_am2, -1, va_sm2)


def _va_primary_to_va_global(coords: NDArray[np.float32]) -> NDArray[np.float32]:
    return _va_mirror_to_global(coords, va_am1, 1, va_sm1)


def _va_secondary_to_va_global(coords: NDArray[np.float32]) -> NDArray[np.float32]:
    return _va_mirror_to_global(coords, va_am2, -1, va_sm2)


def _va_primary_to_va_secondary(coords: NDArray[np.float32]) -> NDArray[np.float32]:
    return _va_global_to_mirror(
        _va_mirror_to_global(coords, va_am1, 1, va_sm1), va_am2, -1, va_sm2
    )


def _va_secondary_to_va_primary(coords: NDArray[np.float32]) -> NDArray[np.float32]:
    return _va_global_to_mirror(
        _va_mirror_to_global(coords, va_am2, 1, va_sm2), va_am1, 1, va_sm1
    )


def _opt_global_to_va_global(coords: NDArray[np.float32]) -> NDArray[np.float32]:
    coords_transformed = coords[:, [1, 0, 2]]
    coords_transformed[:, 1] *= -1
    coords_transformed = coords + vg2og_shift
    return coords_transformed


def _opt_global_to_va_primary(coords: NDArray[np.float32]) -> NDArray[np.float32]:
    return _va_global_to_va_primary(_opt_global_to_va_global(coords))


def _opt_global_to_va_secondary(coords: NDArray[np.float32]) -> NDArray[np.float32]:
    return _va_global_to_va_secondary(_opt_global_to_va_global(coords))


def _opt_primary_to_va_global(coords: NDArray[np.float32]) -> NDArray[np.float32]:
    return _opt_global_to_va_global(_opt_primary_to_opt_global(coords))


def _opt_primary_to_va_primary(coords: NDArray[np.float32]) -> NDArray[np.float32]:
    return _opt_global_to_va_primary(_opt_primary_to_opt_global(coords))


def _opt_primary_to_va_secondary(coords: NDArray[np.float32]) -> NDArray[np.float32]:
    return _opt_global_to_va_secondary(_opt_primary_to_opt_global(coords))


def _opt_secondary_to_va_global(coords: NDArray[np.float32]) -> NDArray[np.float32]:
    return _opt_global_to_va_global(_opt_secondary_to_opt_global(coords))


def _opt_secondary_to_va_primary(coords: NDArray[np.float32]) -> NDArray[np.float32]:
    return _opt_global_to_va_primary(_opt_secondary_to_opt_global(coords))


def _opt_secondary_to_va_secondary(coords: NDArray[np.float32]) -> NDArray[np.float32]:
    return _opt_global_to_va_secondary(_opt_secondary_to_opt_global(coords))


def _va_global_to_opt_global(coords: NDArray[np.float32]) -> NDArray[np.float32]:
    coords_transformed = coords - vg2og_shift
    coords_transformed = coords_transformed[:, [1, 0, 2]]
    coords_transformed[:, 2] *= -1
    return coords_transformed


def _va_global_to_opt_primary(coords: NDArray[np.float32]) -> NDArray[np.float32]:
    # return _opt_global_to_opt_primary(_va_global_to_opt_global(coords))
    coords_transformed = _va_global_to_va_primary(coords)
    coords_transformed = coords_transformed[:, [1, 0, 2]]
    coords_transformed[:, 2] *= -1
    return coords_transformed


def _va_global_to_opt_secondary(coords: NDArray[np.float32]) -> NDArray[np.float32]:
    return _opt_global_to_opt_secondary(_va_global_to_opt_global(coords))


def _va_primary_to_opt_global(coords: NDArray[np.float32]) -> NDArray[np.float32]:
    return _va_global_to_opt_global(_va_primary_to_va_global(coords))


def _va_primary_to_opt_primary(coords: NDArray[np.float32]) -> NDArray[np.float32]:
    return _va_global_to_opt_primary(_va_primary_to_va_global(coords))


def _va_primary_to_opt_secondary(coords: NDArray[np.float32]) -> NDArray[np.float32]:
    return _va_global_to_opt_secondary(_va_primary_to_va_global(coords))


def _va_secondary_to_opt_global(coords: NDArray[np.float32]) -> NDArray[np.float32]:
    return _va_global_to_opt_global(_va_secondary_to_va_global(coords))


def _va_secondary_to_opt_primary(coords: NDArray[np.float32]) -> NDArray[np.float32]:
    return _va_global_to_opt_primary(_va_secondary_to_va_global(coords))


def _va_secondary_to_opt_secondary(coords: NDArray[np.float32]) -> NDArray[np.float32]:
    return _va_global_to_opt_secondary(_va_secondary_to_va_global(coords))


def coord_transform(
    coords: NDArray[np.float32], cfrom: str, cto: str
) -> NDArray[np.float32]:
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
    coords : NDArray[np.float32]
        Coordinates to transform.
        Should be a `(npoint, 3)` array.
    cfrom : str
        The coordinate system that `coords` is currently in.
    cto : str
        The coordinate system to put `coords` into.

    Returns
    -------
    coords_transformed : NDArray[np.float32]
        `coords` transformed into `cto`.
    """
    if cfrom == cto:
        return coords
    match f"{cfrom}-{cto}":
        case "opt_global-opt_primary":
            return _opt_global_to_opt_primary(coords)
        case "opt_global-opt_secondary":
            return _opt_global_to_opt_secondary(coords)
        case "opt_primary-opt_global":
            return _opt_primary_to_opt_global(coords)
        case "opt_secondary-opt_global":
            return _opt_secondary_to_opt_global(coords)
        case "opt_primary-opt_secondary":
            return _opt_primary_to_opt_secondary(coords)
        case "opt_secondary-opt_primary":
            return _opt_secondary_to_opt_primary(coords)
        case "va_global-va_primary":
            return _va_global_to_va_primary(coords)
        case "va_global-va_secondary":
            return _va_global_to_va_secondary(coords)
        case "va_primary-va_global":
            return _va_primary_to_va_global(coords)
        case "va_secondary-va_global":
            return _va_secondary_to_va_global(coords)
        case "va_primary-va_secondary":
            return _va_primary_to_va_secondary(coords)
        case "va_secondary-va_primary":
            return _va_secondary_to_va_primary(coords)
        case "opt_global-va_global":
            return _opt_global_to_va_global(coords)
        case "opt_global-va_primary":
            return _opt_global_to_va_primary(coords)
        case "opt_global-va_secondary":
            return _opt_global_to_va_secondary(coords)
        case "opt_primary-va_global":
            return _opt_primary_to_va_global(coords)
        case "opt_primary-va_primary":
            return _opt_primary_to_va_primary(coords)
        case "opt_primary-va_secondary":
            return _opt_primary_to_va_secondary(coords)
        case "opt_secondary-va_global":
            return _opt_secondary_to_va_global(coords)
        case "opt_secondary-va_primary":
            return _opt_secondary_to_va_primary(coords)
        case "opt_secondary-va_secondary":
            return _opt_secondary_to_va_secondary(coords)
        case "va_global-opt_global":
            return _va_global_to_opt_global(coords)
        case "va_global-opt_primary":
            return _va_global_to_opt_primary(coords)
        case "va_global-opt_secondary":
            return _va_global_to_opt_secondary(coords)
        case "va_primary-opt_global":
            return _va_primary_to_opt_global(coords)
        case "va_primary-opt_primary":
            return _va_primary_to_opt_primary(coords)
        case "va_primary-opt_secondary":
            return _va_primary_to_opt_secondary(coords)
        case "va_secondary-opt_global":
            return _va_secondary_to_opt_global(coords)
        case "va_secondary-opt_primary":
            return _va_secondary_to_opt_primary(coords)
        case "va_secondary-opt_secondary":
            return _va_secondary_to_opt_secondary(coords)
        case _:
            raise ValueError("Invalid coordinate system provided!")


def affine_basis_transform(
    aff: NDArray[np.float32],
    sft: NDArray[np.float32],
    cfrom: str,
    cto: str,
    src_or_dst: bool = True,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """
    Take an affine transform defined in one coordinate system and move it to another.
    The valid coordinate systems are the same as in `coord_transform`.

    Parameters
    ----------
    aff : NDArray[np.float32]
        Affine matrix to tranform.
        Should be a `(3, 3)` array.
    sft : NDArray[np.float32]
        Shift vector to tranform.
        Should be a `(3,)` array.
    cfrom : str
        The coordinate system that `aff` and `sft` is currently in.
    cto : str
        The coordinate system to put `aff` and `sft` into.
    src_or_dst : bool, default: True
        If `True` then the coordinate transform is done on the source
        points that we are affine transforming.
        This is equivalent to doing `aff@(coord_transform(src)) + sft`.
        If `False` then the coordinate transform is done on the destination
        points obtained by the affine transform.
        This is equivalent to doing `coord_transform(aff@src + sft)`

    Returns
    -------
    aff_transformed : NDArray[np.float32]
        `aff` transformed into `cto`.
    sft_transformed : NDArray[np.float32]
        `sft` transformed into `cto`.
    """
    # Make a grid of reference points
    line = np.array((-1, 0, 1), np.float32)
    x, y, z = np.meshgrid(line, line, line)
    xyz = np.column_stack((x.ravel(), y.ravel(), z.ravel()))

    # Apply the affine transform
    xyz_transformed = apply_transform(xyz, aff, sft)

    # Move to the new coordinate system
    if src_or_dst:
        xyz = coord_transform(xyz, cfrom, cto)
    else:
        xyz_transformed = coord_transform(xyz_transformed, cfrom, cto)

    # Get the new affine transform
    aff, sft = get_affine(xyz, xyz_transformed)

    return aff, sft


def align_photo(
    labels: NDArray[np.str_],
    coords: NDArray[np.float32],
    reference: dict,
    *,
    plot: bool = True,
    mirror: str = "primary",
    max_dist: float = 100.0,
) -> tuple[
    NDArray[np.str_],
    NDArray[np.float32],
    NDArray[np.bool_],
    tuple[NDArray[np.float32], NDArray[np.float32]],
]:
    """
    Align photogrammetry data and then put it into mirror coordinates.

    Parameters
    ----------
    labels : NDArray[np.str_]
        The labels of each photogrammetry point.
        Should have shape `(npoint,)`.
    coords : NDArray[np.float32]
        The coordinates of each photogrammetry point.
        Should have shape `(npoint, 3)`.
    reference : dict
        Reference dictionary.
        Should contain a key called `coords` that specifies the
        coordinate system that the reference points are in.
        The rest of the keys should be optical elements (ie: "primary")
        pointing to a list of reference points to use.
        Each point given should be a tuple with two elements.
        The first element is a tuple with the (x, y, z) coordinates
        of the point in the global coordinate system.
        The second is a list of nearby coded targets that can be used
        to identify the point.
    plot : bool, default: True
        If True show a diagnostic plot of how well the reference points
        are aligned.
    mirror : str, default: 'primary'
        The mirror that these points belong to.
        Should be either: 'primary' or 'secondary'.
    max_dist : float, default: 100
        Max distance in mm that the reference poing can be from the target
        point used to locate it.

    Returns
    -------
    labels : NDArray[np.str_]
        The labels of each photogrammetry point.
        Invar points are not included.
    coords_transformed : NDArray[np.float32]
        The transformed coordinates.
        Invar points are not included.
    msk : NDArray[np.bool_]
        Mask to removes invar points
    alignment : tuple[NDArray[np.float32], NDArray[np.float32]]
        The transformation that aligned the points.
        The first element is a rotation matrix and
        the second is the shift.
    """
    logger.info("\tAligning with reference points for %s", mirror)
    if mirror not in ["primary", "secondary"]:
        raise ValueError(f"Invalid mirror: {mirror}")
    if len(reference) == 0:
        raise ValueError("Invalid or empty reference")
    if mirror not in reference:
        raise ValueError("Mirror not found in reference dict")
    if "coords" not in reference:
        raise ValueError("Reference coordinate system not specified")
    if mirror == "primary":
        transform = partial(
            coord_transform, cfrom=reference["coords"], cto="opt_primary"
        )
    else:
        transform = partial(
            coord_transform, cfrom=reference["coords"], cto="opt_secondary"
        )

    # Lets find the points we can use
    trg_idx = np.where(np.char.find(labels, "TARGET") >= 0)[0]
    ref = []
    pts = []
    invars = []
    for rpoint, codes in reference[mirror]:
        have = np.isin(codes, labels)
        if np.sum(have) == 0:
            continue
        coded = coords[np.where(labels == codes[np.where(have)[0][0]])[0][0]]
        # Find the closest point
        dist = np.linalg.norm(coords[trg_idx] - coded, axis=-1)
        if np.min(dist) > max_dist:
            continue
        ref += [rpoint]
        pts += [coords[trg_idx][np.argmin(dist)]]
        invars += [labels[trg_idx][np.argmin(dist)]]
    if len(ref) < 4:
        raise ValueError(f"Only {len(ref)} reference points found! Can't align!")
    logger.debug(
        "\t\tFound %d reference points in measurements with labels:\n\t\t\t%s",
        len(pts),
        str(invars),
    )
    pts = np.vstack(pts)
    ref = np.vstack(ref)
    pts = np.vstack((pts, np.mean(pts, 0)))
    ref = np.vstack((ref, np.mean(ref, 0)))
    ref = transform(ref)
    logger.debug("\t\tReference points in mirror coords:\n%s", str(ref[:-1]))
    triu_idx = np.triu_indices(len(pts), 1)
    scale_fac = np.nanmedian(make_edm(ref)[triu_idx] / make_edm(pts)[triu_idx])
    logger.debug("\t\tScale factor of %f applied", scale_fac)
    pts *= scale_fac

    rot, sft = get_rigid(pts, ref, method="mean")
    pts_t = apply_transform(pts, rot, sft)

    if plot:
        plt.scatter(pts_t[:, 0], pts_t[:, 1], color="b")
        plt.scatter(ref[:, 0], ref[:, 1], color="r")
        plt.show()
    logger.info(
        "\t\tRMS of reference points after alignment: %f",
        np.sqrt(np.mean((pts_t - ref) ** 2)),
    )
    coords_transformed = apply_transform(coords, rot, sft)

    msk = ~np.isin(labels, invars)

    return labels[msk], coords_transformed[msk], msk, (rot, sft)
