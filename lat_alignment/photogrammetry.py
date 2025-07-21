"""
Code for handling and processing photogrammetry data
"""

import logging
from copy import deepcopy
from dataclasses import dataclass
from functools import cached_property, partial
from typing import Self

import matplotlib.pyplot as plt
import numpy as np
from megham.transform import apply_transform, decompose_rotation, get_affine, get_rigid
from megham.utils import make_edm
from numpy.typing import NDArray

from .transforms import coord_transform

logger = logging.getLogger("lat_alignment")


@dataclass
class Dataset:
    """
    Container class for photogrammetry dataset.
    Provides a dict like interface for accessing points by their labels.

    Attributes
    ----------
    data_dict : dict[str, NDArray[np.float64]]
        Dict of photogrammetry points.
        You should genrally not touch this directly.
    """

    data_dict: dict[str, NDArray[np.float64]]

    def _clear_cache(self):
        self.__dict__.pop("points", None)
        self.__dict__.pop("labels", None)
        self.__dict__.pop("codes", None)
        self.__dict__.pop("code_labels", None)
        self.__dict__.pop("target", None)
        self.__dict__.pop("target_labels", None)

    def __setattr__(self, name, value):
        if name == "data_dict":
            self._clear_cache()
        return super().__setattr__(name, value)

    def __setitem__(self, key, item):
        self._clear_cache()
        self.data_dict[key] = item

    def __getitem__(self, key):
        return self.data_dict[key]

    def __repr__(self):
        return repr(self.data_dict)

    def __len__(self):
        return len(self.data_dict)

    def __delitem__(self, key):
        self._clear_cache()
        del self.data_dict[key]

    def __contains__(self, item):
        return item in self.data_dict

    def __iter__(self):
        return iter(self.data_dict)

    @cached_property
    def points(self) -> NDArray[np.float64]:
        """
        Get all points in the dataset as an array.
        This is cached.
        """
        return np.array(list(self.data_dict.values()))

    @cached_property
    def labels(self) -> NDArray[np.str_]:
        """
        Get all labels in the dataset as an array.
        This is cached.
        """
        return np.array(list(self.data_dict.keys()))

    @cached_property
    def codes(self) -> NDArray[np.float64]:
        """
        Get all coded points in the dataset as an array.
        This is cached.
        """
        msk = np.char.find(self.labels, "CODE") >= 0
        return self.points[msk]

    @cached_property
    def code_labels(self) -> NDArray[np.str_]:
        """
        Get all coded labels in the dataset as an array.
        This is cached.
        """
        msk = np.char.find(self.labels, "CODE") >= 0
        return self.labels[msk]

    @cached_property
    def targets(self) -> NDArray[np.float64]:
        """
        Get all target points in the dataset as an array.
        This is cached.
        """
        msk = np.char.find(self.labels, "TARGET") >= 0
        return self.points[msk]

    @cached_property
    def target_labels(self) -> NDArray[np.str_]:
        """
        Get all target labels in the dataset as an array.
        This is cached.
        """
        msk = np.char.find(self.labels, "TARGET") >= 0
        return self.labels[msk]

    def copy(self) -> Self:
        """
        Make a deep copy of the dataset.

        Returns
        -------
        copy : Dataset
            A deep copy of this dataset.
        """
        return deepcopy(self)


# def _blind_search(dataset: Dataset, refs: NDArray[np.float64], found: list[str], which: list[int], tol: float):
#     if len(found) == 0:
#         raise ValueError("Cannot do blind search with zero located targets")
#     edm_ref = make_edm(refs)
#     edm = make_edm(dataset.targets)


def align_photo(
    dataset: Dataset,
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
    Dataset,
    tuple[NDArray[np.float64], NDArray[np.float64]],
]:
    """
    Align photogrammetry data and then put it into mirror coordinates.

    Parameters
    ----------
    dataset : Dataset
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
    aligned : Dataset
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
    transformed = Dataset(data)

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
