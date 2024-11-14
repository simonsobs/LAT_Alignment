"""
Calculate adjustments needed to align LAT mirror panel

Author: Saianeesh Keshav Haridas
"""

from typing import Tuple

import numpy as np
import scipy.optimize as opt
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as rot


def rotate(
    point: NDArray[np.float32],
    end_point1: NDArray[np.float32],
    end_point2: NDArray[np.float32],
    theta: np.float32,
) -> NDArray[np.float32]:
    """
    Rotate a point about an axis

    Parameters
    ----------
    point : NDArray[np.float32]
        The point to rotate
    end_point1 : NDArray[np.float32]
        A point on the axis of rotation
    end_point2 : NDArray[np.float32]
        Another point on the axis of rotation
    theta: NDArray[np.float32]
        Angle in radians to rotate by

    Returns
    -------
    point : NDArray[np.float32]
        The rotated point
    """
    origin = np.mean((end_point1, end_point2))
    point_0 = point - origin
    ax = end_point2 - end_point1
    ax = rot.from_rotvec(theta * ax / np.linalg.norm(ax))
    point_0 = ax.apply(point_0)
    return point_0 + origin


def rotate_panel(
    points: NDArray[np.float32],
    adjustors: NDArray[np.float32],
    thetha_0: np.float32,
    thetha_1: np.float32,
) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
    """
    Rotate panel about axes created by adjustors.

    Parameters
    ----------
    points : NDArray[np.float32]
        Points on panel to rotate.
    adjustors : NDArray[np.float32]
        Adjustor positions.
    thetha_0 : np.float32
        Angle to rotate about first adjustor axis
    thetha_1 : np.float32.
        Angle to rotate about second adjustor axis

    Returns
    -------
    rot_points : NDArray[np.float32]
        The rotated points.
    rot_adjustors : NDArray[np.float32]
        The rotated adjustors.
    """
    rot_points = np.zeros(points.shape, np.float32)
    rot_adjustors = np.zeros(adjustors.shape, np.float32)

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
    points: NDArray[np.float32],
    adjustors: NDArray[np.float32],
    dx: np.float32,
    dy: np.float32,
    dz: np.float32,
) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
    """
    Translate a panel.

    Parameters
    ----------
    points : NDArray[np.float32]
        The points on panel to translate.
    adjustors : NDArray[np.float32]
        Adjustor positions.
    dx : np.float32
        Translation in x.
    dy : np.float32
        Translation in y.
    dz : np.float32
        Translation in z.

    Returns
    -------
    points : NDArray[np.float32]
        The translated points.
    adjustors : NDArray[np.float32]
        The translated adjustors.
    """
    translation = np.array((dx, dy, dz))
    return points + translation, adjustors + translation


def adjustment_fit_func(
    pars: NDArray[np.float32],
    can_points: NDArray[np.float32],
    points: NDArray[np.float32],
    adjustors: NDArray[np.float32],
) -> np.float32:
    r"""
    Function to minimize when calculating adjustments.

    Parameters
    ----------
    pars : NDArray[np.float32]
        The parameters to fit for:

        * dx: Translation in x
        * dy: Translation in y
        * dz: Translation in z
        * thetha_0: Angle to rotate about first adjustor axis
        * thetha_1: Angle to rotate about second adjustor axis
        * z_t: Additional translation to tension the center point
    can_points : NDArray[np.float32]
        The cannonical positions of the points to align.
    points : NDArray[np.float32]
        The measured positions of the points to align.
    adjustors : NDArray[np.float32]
        The measured positions of the adjustors.

    Returns
    -------
    norm : np.float32
        The norm of $cannonical_positions - transformed_positions$.
    """
    dx, dy, dz, thetha_0, thetha_1, z_t = pars
    points, adjustors = translate_panel(points, adjustors, dx, dy, dz)
    points, adjustors = rotate_panel(points, adjustors, thetha_0, thetha_1)
    points[-1, -1] += z_t
    return np.linalg.norm(can_points - points)


def calc_adjustments(
    can_points: NDArray[np.float32],
    points: NDArray[np.float32],
    adjustors: NDArray[np.float32],
    **kwargs,
) -> Tuple[
    np.float32,
    np.float32,
    NDArray[np.float32],
    np.float32,
    np.float32,
    NDArray[np.float32],
]:
    """
    Calculate adjustments needed to align panel.

    Parameters
    ----------
    can_points : NDArray[np.float32]
        The cannonical position of the points to align.
    points : NDArray[np.float32]
        The measured positions of the points to align.
    adjustors : NDArray[np.float32]
        The measured positions of the adjustors.
    **kwargs
        Arguments to be passed to `scipy.optimize.minimize`.

    dx : np.float32
        The required translation of panel in x.
    dy : np.float32
        The required translation of panel in y.
    d_adj : NDArray[np.float32]
        The amount to move each adjustor.
    dx_err : np.float32
        The error in the fit for `dx`.
    dy_err : np.float32
        The error in the fit for `dy`.
    d_adj_err : NDArray[np.float32]
        The error in the fit for `d_adj`.
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
