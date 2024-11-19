"""
Module for fitting for the bearing location.
Needs to be made less hardcoded...
"""
import logging
import cylinder_fitting as cf
import numpy as np
from numpy.typing import NDArray
from skspatial.objects import Line, Vector, Plane
from scipy.spatial.transform import Rotation as R
from megham.transform import decompose_rotation

logger = logging.getLogger("lat_alignment")

# On the same plane as the face facing the receiver
ORIGIN = np.array([0., 4510., 0.]) # opt_global
# ZERO = np.array([-1.33556967e+00,  4.42348732e+03,  2.00727038e+03])
ZERO = np.array([-1.63167553e+00,  4.42358440e+03,  2.00653089e+03])
# ZERO_CODED = np.array([  48.14332494, 4426.74247708, 2006.50467441])
ZERO_CODED = np.array([  47.9347961 , 4426.79014228, 2005.81459522])
ZERO_CODE = "CODE90"
AXIS1 = Vector.from_points(ORIGIN, ORIGIN + np.array([0., -1., 0.])).unit()
AXIS2 = Vector.from_points(ORIGIN, ZERO).unit()
AXIS2_CODED = Vector.from_points(ORIGIN, ZERO_CODED).unit()

def partition_points(data: dict[str, NDArray[np.float32]], coded: dict[str, NDArray[np.float32]]) -> tuple[NDArray[np.float32], NDArray[np.float32], bool]:
    coords = np.array(list(data.values()))

    if ZERO_CODE not in coded:
        raise ValueError("Can't find zero point of bearing! Coded target not found!")
    zero_coded = coded[ZERO_CODE]
    dist = np.linalg.norm(coords - zero_coded, axis=-1)
    zero_point = coords[np.argmin(dist)]
    is_code = False
    if np.min(dist) > 100:
        logger.error("\t\tCan't find zero point of bearing! Falling back to coded target!")
        zero_point = zero_coded
        is_code = True
    print(zero_point)

    inside_msk = (coords[:, 1] > ORIGIN[1] - 100) * (coords[:, 1] < ORIGIN[1] - 10)
    inside_points = coords[inside_msk]
    if len(inside_points) < 4:
        raise ValueError("Not enough points on inner bearing surface found!")

    return inside_points, zero_point, is_code

def cylinder_fit(data: dict[str, NDArray[np.float32]], coded: dict[str, NDArray[np.float32]]) -> tuple[NDArray[np.float32], tuple[NDArray[np.float32], NDArray[np.float32]]]:
    # Partition points
    logger.info("\tStarting fit of bearing axis")
    inside_points, zero_point, is_code = partition_points(data, coded)
    logger.info("\t\tFound %d points on bearing surface", len(inside_points))
    logger.debug("\t\tZero point is at %s", str(zero_point))

    # Fit inside of bearing
    w, c, *_ = cf.fit(inside_points)
    center_line = Line(point=c, direction=w)
    
    # Fit face of bearing
    face_points = np.array([coded[label] for label in ["CODE91", "CODE92", "CODE93", "CODE94"]])
    face_plane = Plane.best_fit(face_points)
    origin = face_plane.intersect_line(center_line)
    shift = ORIGIN - origin

    # Get our basis
    axis1 = Vector.from_points(origin + shift, center_line.point + shift).unit()
    axis2 = Vector.from_points(origin + shift, zero_point + shift).unit()

    # Get the transform that aligns us 
    _AXIS2 = AXIS2
    if is_code:
        _AXIS2 = AXIS2_CODED
    rot, err, *_ = R.align_vectors(np.vstack((axis1, axis2)), np.vstack((AXIS1, _AXIS2)))
    logger.info("\t\tError on bearing axis alignment: %f", err)

    shift = rot.apply(shift)
    rot = np.array(rot.as_matrix(), dtype=np.float32)
    logger.debug("\t\tShift is %s mm", str(shift))
    logger.debug("\t\tRotation is %s deg", str(np.rad2deg(decompose_rotation(rot))))

    return inside_points, (rot, shift)
