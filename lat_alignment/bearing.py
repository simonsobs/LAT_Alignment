"""
Module for fitting for the bearing location.
Needs to be made less hardcoded...
"""

import logging

import cylinder_fitting as cf
import numpy as np
from megham.transform import apply_transform, decompose_rotation
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as R
from skspatial.objects import Line, Plane, Vector

from .photogrammetry import Dataset

logger = logging.getLogger("lat_alignment")

# On the same plane as the face facing the receiver
# ORIGIN = np.array([0.0, 4366.0, 0.0])  # opt_global
# ZERO = np.array([5.66309902, 4279.33935856, 2001.09289888]) 
# ZERO_CODED = np.array([55.12701416, 4282.49464202, 1999.9269164]) 
# FACE_TOL = 5
ORIGIN = np.array([0.0, 4410.0, 0.0])  # opt_global
ZERO = np.array([-1.33556967e00, 4.32348732e03, 2.00727038e03])
ZERO_CODED = np.array([48.14332494, 4326.74247708, 2006.50467441])
FACE_TOL = .5
ZERO_CODE = "CODE90"
AXIS1 = Vector.from_points(ORIGIN, ORIGIN + np.array([0.0, -1.0, 0.0])).unit()
AXIS2 = Vector.from_points(ORIGIN, ZERO).unit()
AXIS2_CODED = Vector.from_points(ORIGIN, ZERO_CODED).unit()


def partition_points(
    dataset: Dataset,
) -> tuple[Dataset, Dataset, NDArray[np.float32], NDArray[np.float32]]:
    """
    Split up dataset into points on the bearing reference surface and inner surface.
    Also pulls out the bearing zero points.

    Parameters
    ----------
    dataset : Dataset
        Photogrammetry dataset.
        Should already be aligned to the bearing referance points.

    Returns
    -------
    inside_points : Dataset
        Points on the inner surface of the bearing.
        Only includes targets, codes are removed.
    face_points : Dataset
        Points on face of the bearing that we use as a reference surface.
        Only includes targets, codes are removed.
    zero_point : NDArray[np.float32]
        Array of size (3,) that gives the coordinates of the target we treat
        as the bearing's zero point.
    zero_code: NDArray[np.float32]
        Array of size (3,) that gives the coordinates of the coded target we use
        to identify the bearing's zero point.

    Raises
    ------
    ValueError
        When the zero point of the bearing is not found or there are less than four
        points found on the inner surface or face fo the bearing.
    """
    if ZERO_CODE not in dataset.code_labels:
        raise ValueError("Can't find zero point of bearing! Coded target not found!")
    zero_coded = dataset[ZERO_CODE]
    dist = np.linalg.norm(dataset.targets - zero_coded, axis=-1)
    zero_point = dataset.targets[np.argmin(dist)]
    if np.min(dist) > 100:
        raise ValueError("Can't find zero point of bearing!")

    inside_msk = (dataset.targets[:, 1] > ORIGIN[1] - 100) * (
        dataset.targets[:, 1] < ORIGIN[1] - 10
    )
    inside_points = Dataset(
        {
            l: p
            for l, p in zip(
                dataset.target_labels[inside_msk], dataset.targets[inside_msk]
            )
        }
    )
    if len(inside_points) < 4:
        raise ValueError("Not enough points on inner bearing surface found!")

    face_origin = np.mean(
        np.vstack(
            [
                dataset[label]
                for label in ["CODE91", "CODE92", "CODE93", "CODE94"]
                if label in dataset
            ]
        ),
        axis=0,
    )
    face_msk = (dataset.points[:, 1] > face_origin[1] - FACE_TOL) * (
        dataset.points[:, 1] < face_origin[1] + FACE_TOL
    )
    face_points = Dataset(
        {l: p for l, p in zip(dataset.labels[face_msk], dataset.points[face_msk])}
    )
    if len(face_points) < 4:
        raise ValueError("Not enough points on bearing face surface found!")

    return inside_points, face_points, zero_point, zero_coded


def cylinder_fit(
    dataset: Dataset,
) -> tuple[Dataset, tuple[NDArray[np.float32], NDArray[np.float32]]]:
    """
    Fit for the bearing's position by fitting a cylinder to the bearing surface.
    This acts as a correction on top of the alignment to reference points.

    Parameters
    ----------
    dataset : Dataset
        Photogrammetry dataset.
        Should already be aligned to the bearing referance points.

    Returns
    -------
    inside_points : Dataset
        Points on the inner surface of the bearing with alignment applied.
        Only includes targets, codes are removed.
    alignment : tuple[NDArray[np.float32], NDArray[np.float32]]
        The transformation that aligned the bearing.
        The first element is a rotation matrix and
        the second is the shift.
    """
    # Partition points
    logger.info("\tStarting fit of bearing axis")
    inside_points, face_points, zero_point, zero_coded = partition_points(dataset)
    logger.info("\t\tFound %d points on bearing surface", len(inside_points))
    logger.debug("\t\tZero point is at %s", str(zero_point))

    # Fit inside of bearing
    w, c, *_ = cf.fit(inside_points.points)
    center_line = Line(point=c, direction=w)

    # Fit face of bearing
    face_plane = Plane.best_fit(face_points.points)
    origin = face_plane.intersect_line(center_line)
    shift = ORIGIN - origin

    # Get our basis
    axis1 = Vector.from_points(origin + shift, center_line.point + shift).unit()
    axis2 = Vector.from_points(origin + shift, zero_point + shift).unit()
    axis2_coded = Vector.from_points(origin + shift, zero_coded + shift).unit()

    # Get the transform that aligns us
    rot, err, *_ = R.align_vectors(
        np.vstack((axis1, axis2, axis2_coded)), np.vstack((AXIS1, AXIS2, AXIS2_CODED))
    )
    logger.info("\t\tError on bearing axis alignment: %f", err)

    shift = rot.apply(shift)
    rot = np.array(rot.as_matrix(), dtype=np.float32).T
    logger.debug("\t\tShift is %s mm", str(shift))
    logger.debug("\t\tRotation is %s deg", str(np.rad2deg(decompose_rotation(rot))))

    coords_transformed = apply_transform(inside_points.points, rot, shift)
    data = {
        label: coord for label, coord in zip(inside_points.labels, coords_transformed)
    }

    return Dataset(data), (rot, shift)
