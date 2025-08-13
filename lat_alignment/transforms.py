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

import numpy as np
from megham.transform import apply_transform, decompose_rotation, get_affine, get_rigid
from megham.utils import make_edm
import matplotlib.pyplot as plt
from numpy.typing import NDArray

from .dataset import DatasetPhotogrammetry

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


# def _blind_search(dataset: DatasetPhotogrammetry, refs: NDArray[np.float64], found: list[str], which: list[int], tol: float):
#     if len(found) == 0:
#         raise ValueError("Cannot do blind search with zero located targets")
#     edm_ref = make_edm(refs)
#     edm = make_edm(dataset.targets)


def align_photo(
    dataset: DatasetPhotogrammetry,
    reference: dict,
    kill_refs: bool,
    element: str = "primary",
    scale: bool = True,
    blind_search: float = -1,
    *,
    plot: bool = True,
    max_dist: float = 100.0,
    rms_thresh: float = 1.0,
) -> tuple[
    DatasetPhotogrammetry,
    tuple[NDArray[np.float64], NDArray[np.float64]],
]:
    """
    Align photogrammetry data and then put it into mirror coordinates.

    Parameters
    ----------
    dataset : DatasetPhotogrammetry
        The photogrammetry data to align.
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
        Each item in the list of coded targets should be a tuple containing
        the label of the code and the (x, y, z) coordinate of the coded target.
    kill_refs : bool
        If True remove reference points from the dataset.
    element : str, default: 'primary'
        The element that these points belong to.
        Should be either: 'primary', 'secondary', 'bearing', 'receiver', or 'all'.
    scale : bool, default: True
        If True also compute a scale factor from the reference points.
    blind_search : float, default: -1
        Perform a blind search for the reference points.
        This is not implemented yet...
    plot : bool, default: True
        If True show a diagnostic plot of how well the reference points
        are aligned.
    max_dist : float, default: 100
        Max distance in mm that the reference poing can be from the target
        point used to locate it.
    rms_thresh : float, default: 1
        RMS is mm above which we will attempt to cut points.

    Returns
    -------
    aligned : DatasetPhotogrammetry
        The photogrammetry data aligned to the reference points.
    alignment : tuple[NDArray[np.float64], NDArray[np.float64]]
        The transformation that aligned the points.
        The first element is a rotation matrix and
        the second is the shift.
    """
    logger.info("\tAligning with reference points for %s", element)
    elements = ["primary", "secondary", "bearing", "receiver"]
    # import ipdb; ipdb.set_trace()
    if element not in elements and element != "all":
        raise ValueError(f"Invalid element: {element}")
    if len(reference) == 0:
        raise ValueError("Invalid or empty reference")
    if element not in reference and element != "all":
        raise ValueError("Element not found in reference dict")
    if "coords" not in reference:
        raise ValueError("Reference coordinate system not specified")
    if element == "primary":
        transform = partial(
            coord_transform, cfrom=reference["coords"], cto="opt_primary"
        )
    elif element == "secondary":
        transform = partial(
            coord_transform, cfrom=reference["coords"], cto="opt_secondary"
        )
    else:
        transform = partial(
            coord_transform, cfrom=reference["coords"], cto="opt_global"
        )
    if element == "all":
        all_refs = []
        for el in elements:
            if el not in reference:
                continue
            all_refs += reference[el]
        reference["all"] = all_refs

    # Lets find the points we can use
    ref = []
    pts = []
    invars = []
    ref_coded = []
    found_coded = []
    for rpoint, codes in reference[element]:
        code_l = np.array([l for l, _ in codes])
        code_p = np.array([p for _, p in codes])
        have = np.isin(code_l, dataset.code_labels)
        if np.sum(have) == 0:
            continue
        # Save the coded we have just in case
        ref_coded += [code_p[have]]
        found_coded += [dataset[l] for l in code_l[have]]

        # Use the first found coded as reference
        coded = dataset[code_l[have][0]]
        # Find the closest point
        dist = np.linalg.norm(dataset.targets - coded, axis=-1)
        if np.min(dist) > max_dist:
            continue
        label = dataset.target_labels[np.argmin(dist)]
        ref += [rpoint]
        pts += [dataset[label]]
        invars += [label]
    if blind_search > 0:
        raise NotImplementedError("Blind search not implemented yet!")
    # Set 12
    # ref = [rpoint for rpoint, _ in reference[element]]
    # ref = np.array(ref)[[True, True, False, True]]
    # invars = ["TARGET35", "TARGET4", "TARGET484"] #, "TARGET421"]
    # pts = [dataset[label] for label in invars]
    # print(invars)
    if len(ref) < 3:
        logger.warning(f"Only {len(ref)} reference points found!")
        logger.warning(f"Adding reference codes")
        pts += found_coded
        ref += ref_coded
    if len(ref) < 3:
        raise ValueError(
            f"Only {len(ref)} reference points found including codes! Can't align!"
        )
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
    logger.debug("\t\tReference points in element coords:\n%s", str(ref))

    msk = np.ones(len(ref), bool)
    scale_fac = 1
    rot = None
    sft = None
    rms = np.inf
    for _ in range(len(ref) - 4):
        rot, sft = get_rigid(pts[msk], ref[msk], method="mean")
        if scale:
            triu_idx = np.triu_indices(len(pts[msk]), 1)
            scale_fac = np.nanmedian(
                make_edm(ref[msk])[triu_idx] / make_edm(pts[msk])[triu_idx]
            )
        pts_scaled = pts * scale_fac
        logger.debug("\t\tScale factor of %f applied", scale_fac)

        new_rot, new_sft = get_rigid(pts_scaled[msk], ref[msk], method="mean")
        pts_t = apply_transform(pts_scaled[msk], new_rot, new_sft)
        if plot:
            fig = plt.figure()
            ax = fig.add_subplot(projection="3d")
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], color="g")
            ax.scatter(pts_t[:, 0], pts_t[:, 1], pts_t[:, 2], color="b")
            ax.scatter(ref[:, 0], ref[:, 1], ref[:, 2], color="r", marker="X")
            plt.show()
        diff = pts_t - ref[msk]
        new_rms = np.sqrt(np.mean((diff) ** 2))
        diff = np.linalg.norm(diff, axis=1)
        if new_rms > rms:
            logger.info("\t\tNew RMS is worse, accepting last try")
            break
        rms = new_rms
        rot = new_rot
        sft = new_sft
        logger.info(
            "\t\tRMS of reference points after alignment: %f",
            rms,
        )
        if rms <= rms_thresh:
            break
        logger.info("\t\tRMS over thresh, trying cutting worst point")
        to_cut = np.argmax(np.abs(diff))
        _msk = msk[msk].copy()
        _msk[to_cut] = False
        msk[msk] = _msk

    if rot is None or sft is None:
        raise ValueError("Transformation is None")

    coords_transformed = apply_transform(dataset.points * scale_fac, rot, sft)
    labels = dataset.labels

    if kill_refs:
        msk = ~np.isin(dataset.labels, invars)
        labels = labels[msk]
        coords_transformed = coords_transformed[msk]

    data = {label: coord for label, coord in zip(labels, coords_transformed)}
    transformed = DatasetPhotogrammetry(data)

    logger.debug("\t\tShift is %s mm", str(sft))
    logger.debug("\t\tRotation is %s deg", str(np.rad2deg(decompose_rotation(rot))))
    scale_fac = np.eye(3) * scale_fac
    rot @= scale_fac

    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        ax.scatter(
            transformed.targets[:, 0],
            transformed.targets[:, 1],
            transformed.targets[:, 2],
            marker="x",
        )
        plt.show()

    return transformed, (rot, sft)
