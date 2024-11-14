"""
Calculate adjustments needed to align LAT mirror panel

Author: Saianeesh Keshav Haridas
"""
import numpy as np
import scipy.optimize as opt
from scipy.spatial.transform import Rotation as rot
from numpy import float64, ndarray
from typing import Tuple


def rotate(
    point: ndarray, end_point1: ndarray, end_point2: ndarray, thetha: float64
) -> ndarray:
    """
    Rotate a point about an axis

    @param point: The point to rotate
    @param end_point1: A point on the axis of rotation
    @param end_point2: Another point on the axis of rotation
    @param thetha: Angle in radians to rotate by

    @return point: The rotated point
    """
    origin = np.mean((end_point1, end_point2))
    point_0 = point - origin
    ax = end_point2 - end_point1
    ax = rot.from_rotvec(thetha * ax / np.linalg.norm(ax))
    point_0 = ax.apply(point_0)
    return point_0 + origin


def rotate_panel(
    points: ndarray, adjustors: ndarray, thetha_0: float64, thetha_1: float64
) -> Tuple[ndarray, ndarray]:
    """
    Rotate panel about axes created by adjustors

    @param points: Points on panel to rotate
    @param adjustors: Adjustor positions
    @param thetha_0: Angle to rotate about first adjustor axis
    @param thetha_1: Angle to rotate about second adjustor axis

    @return rot_points: The rotated points
    @return rot_adjustors: The rotated adjustors
    """
    rot_points = np.zeros(points.shape)
    rot_adjustors = np.zeros(adjustors.shape)

    n_points = len(points)
    n_adjustors = len(adjustors)

    for i in range(n_points):
        rot_points[i] = rotate(points[i], adjustors[1], adjustors[2], thetha_0)
        rot_points[i] = rotate(rot_points[i], adjustors[0], adjustors[3], thetha_1)
    for i in range(n_adjustors):
        rot_adjustors[i] = rotate(adjustors[i], adjustors[1], adjustors[2], thetha_0)
        rot_adjustors[i] = rotate(
            rot_adjustors[i], adjustors[0], adjustors[3], thetha_1
        )
    return rot_points, rot_adjustors


def translate_panel(
    points: ndarray, adjustors: ndarray, dx: float64, dy: float64, dz: float64
) -> Tuple[ndarray, ndarray]:
    """
    Translate panel

    @param points: The points on panel to translate
    @param adjustors: Adjustor positions
    @param dx: Translation in x
    @param dy: Translation in y
    @param dz: Translation in z

    @return points: The translated points
    @return adjustors: The translated adjustors
    """
    translation = np.array((dx, dy, dz))
    return points + translation, adjustors + translation


def adjustment_fit_func(
    pars: ndarray, can_points: ndarray, points: ndarray, adjustors: ndarray
) -> float64:
    """
    Function to minimize when calculating adjustments

    @param pars: The parameters to fit for:
                    dx: Translation in x
                    dy: Translation in y
                    dz: Translation in z
                    thetha_0: Angle to rotate about first adjustor axis
                    thetha_1: Angle to rotate about second adjustor axis
                    z_t: Additional translation to tension the center point
    @param can_points: The cannonical positions of the points to align
    @param points: The measured positions of the points to align
    @param adjustors: The measured positions of the adjustors

    @return norm: The norm of (cannonical positions - transformed positions)
    """
    dx, dy, dz, thetha_0, thetha_1, z_t = pars
    points, adjustors = translate_panel(points, adjustors, dx, dy, dz)
    points, adjustors = rotate_panel(points, adjustors, thetha_0, thetha_1)
    points[-1, -1] += z_t
    return np.linalg.norm(can_points - points)


def calc_adjustments(
    can_points: ndarray,
    points: ndarray,
    adjustors: ndarray,
    **kwargs
) -> Tuple[float64, float64, ndarray, float64, float64, ndarray]:
    """
    Calculate adjustments needed to align panel

    @param can_points: The cannonical position of the points to align
    @param points: The measured positions of the points to align
    @param adjustors: The measured positions of the adjustors
    @param **kwargs: Arguments to be passed to scipy.optimize.minimize

    @return dx: The required translation of panel in x
    @return dy: The required translation of panel in y
    @return d_adj: The amount to move each adjustor
    @return dx_err: The error in the fit for dx
    @return dy_err: The error in the fit for dy
    @return d_adj_err: The error in the fit for d_adj
    """
    res = opt.minimize(
        adjustment_fit_func, np.zeros(6), (can_points, points, adjustors), **kwargs
    )

    dx, dy, dz, thetha_0, thetha_1, z_t = res.x
    _points, _adjustors = translate_panel(points, adjustors, dx, dy, dz)
    _points, _adjustors = rotate_panel(_points, _adjustors, thetha_0, thetha_1)
    _adjustors[-1, -1] += z_t
    d_adj = _adjustors - adjustors

    ftol = 2.220446049250313e-09
    if "ftol" in kwargs:
        ftol = kwargs["ftol"]
    perr = np.sqrt(ftol * np.diag(res.hess_inv))
    dx_err, dy_err, dz_err, thetha_0_err, thetha_1_err, z_t_err = perr
    _points, _adjustors = translate_panel(points, adjustors, dx_err, dy_err, dz_err)
    _points, _adjustors = rotate_panel(_points, _adjustors, thetha_0_err, thetha_1_err)
    _adjustors[-1, -1] += z_t_err
    d_adj_err = _adjustors - adjustors

    return dx, dy, d_adj[:, 2], dx_err, dy_err, d_adj_err[:, 2]
