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
from functools import cache, partial
from typing import Optional

import numpy as np
from megham.transform import apply_transform, get_rigid
from megham.utils import make_edm
from numpy.typing import NDArray

# TODO: Write better docs!

DEFAULT_REF = {
    "primary": [
        ((-2818.56, 2400.94, -4819.33), ["CODE14", "CODE15", "CODE28"]),
        ((-2818.83, -2397.25, -4821.03), ["CODE23", "CODE24", "CODE31"]),
        ((2536.61, 2397.31, -2142.25), ["CODE17", "CODE18", "CODE29"]),
        ((2538.4, -2399.23, -2141.8), ["CODE20", "CODE21", "CODE30"]),
    ],
    "secondary": [
        ((-3882.8, -1998.53, 2550.87), ["CODE41", "CODE42", "CODE43"]),
        ((-3883.22, 1993.61, 2551.27), ["CODE32", "CODE33", "CODE34"]),
        ((-5617.53, 1998.7, -2652.13), ["CODE35", "CODE36", "CODE37"]),
        ((-5616.5, -1995.38, -2651.13), ["CODE38", "CODE39", "CODE40"]),
    ],
}


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


def align_photo(
    labels: NDArray[np.str_],
    coords: NDArray[np.float32],
    *,
    mirror: str = "primary",
    reference: Optional[list[tuple[tuple[float, float, float], list[str]]]] = None,
    max_dist: float = 100.0,
) -> tuple[NDArray[np.str_], NDArray[np.float32], NDArray[np.bool_]]:
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
    mirror : str, default: 'primary'
        The mirror that these points belong to.
        Should be either: 'primary' or 'secondary'.
    reference : Optional[list[tuple[tuple[float, float, float], list[str]]]], default: None
        List of reference points to use.
        Each point given should be a tuple with two elements.
        The first element is a tuple with the (x, y, z) coordinates
        of the point in the global coordinate system.
        The second is a list of nearby coded targets that can be used
        to identify the point.
        If `None` the default reference for each mirror is used.
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
    """
    if mirror not in ["primary", "secondary"]:
        raise ValueError(f"Invalid mirror: {mirror}")
    if mirror == "primary":
        transform = partial(coord_transform, cfrom="va_global", cto="opt_primary")
    else:
        transform = partial(coord_transform, cfrom="va_global", cto="opt_secondary")
    if reference is None:
        reference = DEFAULT_REF[mirror]
    if reference is None or len(reference) == 0:
        raise ValueError("Invalid or empty reference")

    # Lets find the points we can use
    trg_idx = np.where(np.char.find(labels, "TARGET") >= 0)[0]
    ref = []
    pts = []
    invars = []
    for rpoint, codes in reference:
        have = np.isin(codes, labels)
        if np.sum(have) == 0:
            continue
        coded = coords[np.where(labels == codes[np.where(have)[0][0]])[0][0]]
        print(codes[np.where(have)[0][0]])
        # Find the closest point
        dist = np.linalg.norm(coords[trg_idx] - coded, axis=-1)
        if np.min(dist) > max_dist:
            continue
        print(np.min(dist))
        ref += [rpoint]
        pts += [coords[trg_idx][np.argmin(dist)]]
        invars += [labels[trg_idx][np.argmin(dist)]]
    if len(ref) < 4:
        raise ValueError(f"Only {len(ref)} reference points found! Can't align!")
    msk = [0, 1, 3]
    pts = np.vstack(pts)[msk]
    ref = np.vstack(ref)[msk]
    pts = np.vstack((pts, np.mean(pts, 0)))
    ref = np.vstack((ref, np.mean(ref, 0)))
    ref = transform(ref)
    print("Reference points in mirror coords:")
    print(ref[:-1])
    print(make_edm(ref) / make_edm(pts))
    print(make_edm(ref) - make_edm(pts))
    print(np.nanmedian(make_edm(ref) / make_edm(pts)))
    pts *= np.nanmedian(make_edm(ref) / make_edm(pts))
    print(make_edm(ref) / make_edm(pts))
    print(make_edm(ref) - make_edm(pts))
    print(np.nanmedian(make_edm(ref) / make_edm(pts)))

    rot, sft = get_rigid(pts, ref, method="mean")
    pts_t = apply_transform(pts, rot, sft)
    import matplotlib.pyplot as plt

    plt.scatter(pts_t[:, 0], pts_t[:, 1], color="b")
    plt.scatter(ref[:, 0], ref[:, 1], color="r")
    plt.show()
    print(pts_t[:-1])
    print(pts_t - ref)
    print(
        f"RMS of reference points after alignment: {np.sqrt(np.mean((pts_t - ref)**2))}"
    )
    coords_transformed = apply_transform(coords, rot, sft)

    msk = ~np.isin(labels, invars)

    return labels[msk], coords_transformed[msk], msk
