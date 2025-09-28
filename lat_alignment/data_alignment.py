"""
Functions for aligning datasets to a well defined reference frame.
"""

import logging
from functools import partial
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from megham.transform import apply_transform, decompose_rotation, get_rigid
from megham.utils import make_edm
from numpy.typing import NDArray

from . import io
from .dataset import Dataset, DatasetPhotogrammetry, DatasetReference
from .transforms import coord_transform, err_transform, err_transform

logger = logging.getLogger("lat_alignment")

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
        all_refs = {}
        for el in elements:
            if el not in reference:
                continue
            all_refs = all_refs | reference[el]
        reference["all"] = all_refs

    # Lets find the points we can use
    ref = []
    pts = []
    invars = []
    ref_coded = []
    found_coded = []
    logger.info("Looking for reference points")
    for pname, (rpoint, codes) in reference[element].items():
        logger.info("\tFinding point %s", pname)
        code_l = np.array([l for l, _ in codes])
        code_p = np.array([p for _, p in codes])
        have = np.isin(code_l, dataset.code_labels)
        logger.info("\t\tFound %d associated codes", np.sum(have))
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
            logger.warning("\t\tFailed to find point %s", pname)
            continue
        label = dataset.target_labels[np.argmin(dist)]
        ref += [rpoint]
        pts += [dataset[label]]
        invars += [label]
        logger.info("\t\tAssociated %s with %s", label, pname)
    if blind_search > 0:
        raise NotImplementedError("Blind search not implemented yet!")
    if len(ref) < 4:
        logger.warning(f"Only {len(ref)} reference points found!")
        logger.warning(f"Adding reference codes")
        pts += found_coded
        ref += ref_coded
    if len(ref) < 4:
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

        new_rot, new_sft = get_rigid(pts_scaled[msk].astype(np.float64), ref[msk], method="mean")
        pts_t = apply_transform(pts_scaled[msk].astype(np.float64), new_rot, new_sft)
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

    logger.debug("\t\tShift is %s mm", str(sft))
    logger.debug("\t\tRotation is %s deg", str(np.rad2deg(decompose_rotation(rot))))
    scale_fac = np.eye(3) * scale_fac
    rot @= scale_fac

    coords_transformed = apply_transform(dataset.points, rot, sft)
    errs_transformed = err_transform(dataset.errs, rot)
    labels = dataset.labels

    if kill_refs:
        msk = ~np.isin(dataset.labels, invars)
        labels = labels[msk]
        coords_transformed = coords_transformed[msk]
        errs_transformed = errs_transformed[msk]

    data = {label: np.array([coord, err]) for label, coord, err in zip(labels, coords_transformed, errs_transformed)}
    transformed = DatasetPhotogrammetry(data)

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


def align_tracker(
    dataset: Dataset,
    tracker_yaml: str,
    element: str = "primary",
    scale: bool = True,
    *,
    plot: bool = True,
) -> tuple[
    Dataset,
    tuple[NDArray[np.float64], NDArray[np.float64]],
]:
    """
    Align photogrammetry data and then put it into mirror coordinates.

    Parameters
    ----------
    dataset : Dataset
        The photogrammetry data to align.
    tracker_yaml : str
        The path to the yaml file with measuerments of the reference points.
    element : str, default: 'primary'
        The element that these points belong to.
        Should be either: 'primary', 'secondary', 'bearing', 'receiver', or 'all'.
    scale : bool, default: True
        If True also compute a scale factor from the reference points.
    plot : bool, default: True
        If True show a diagnostic plot of how well the reference points
        are aligned.

    Returns
    -------
    aligned : Dataset
        The data aligned to the reference points.
    alignment : tuple[NDArray[np.float64], NDArray[np.float64]]
        The transformation that aligned the points.
        The first element is a rotation matrix and
        the second is the shift.
    """
    logger.info("\tAligning with reference points for %s", element)
    elements = ["primary", "secondary", "bearing", "receiver"]
    if element not in elements and element != "all":
        raise ValueError(f"Invalid element: {element}")
    reference = cast(DatasetReference, io.load_tracker(tracker_yaml))
    if len(reference) == 0:
        raise ValueError("Invalid or empty reference")
    if element not in reference.elem_names and element != "all":
        raise ValueError("Element not found in reference dict")
    if element == "primary":
        transform = partial(coord_transform, cfrom="opt_global", cto="opt_primary")
    elif element == "secondary":
        transform = partial(coord_transform, cfrom="opt_global", cto="opt_secondary")
    else:
        transform = partial(coord_transform, cfrom="opt_global", cto="opt_global")
    if element == "all":
        for el in elements:
            if el not in reference.elem_names:
                continue
            for pt in reference.elem_labels[el]:
                reference[f"all_{pt}"] = reference[pt]
                reference[f"all_{pt}_err"] = reference[f"{pt}_err"]
                reference[f"all_{pt}_ref"] = reference[f"{pt}_ref"]

    # Lets find the points we can use
    pts = reference.elements[element]
    ref = reference.reference[element]
    if len(ref) < 3:
        raise ValueError(f"Only {len(ref)} reference points found! Can't align!")
    ref = transform(ref)
    logger.debug("\t\tReference points in element coords:\n%s", str(ref))

    msk = np.ones(len(ref), bool)
    scale_fac = 1
    rot = None
    sft = None
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
    rms = np.sqrt(np.mean((diff) ** 2))
    logger.info(
        "\t\tRMS of reference points after alignment: %f",
        rms,
    )

    logger.debug("\t\tShift is %s mm", str(sft))
    logger.debug("\t\tRotation is %s deg", str(np.rad2deg(decompose_rotation(rot))))
    scale_fac = np.eye(3) * scale_fac
    rot @= scale_fac

    coords_transformed = apply_transform(dataset.points, rot, sft)
    errs_transformed = err_transform(dataset.errs, rot)
    labels = dataset.labels

    data = {label: np.array([coord, err]) for label, coord, err in zip(labels, coords_transformed, errs_transformed)}
    transformed = Dataset(data)

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
