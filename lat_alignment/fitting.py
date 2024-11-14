"""
Functions for fitting against the mirror surface.
"""

import numpy as np
import scipy.optimize as opt
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as rot
from scipy.stats import binned_statistic

from lat_alignment import mirror as mr


def mirror_objective(
    points: NDArray[np.floating], a: NDArray[np.floating], compensate: float = 0
) -> float:
    """
    Objective function to minimize when fitting to mirror surface.
    Essentially just a curvature weighted chisq.

    Parameters
    ----------
    points : NDArray[np.floating]
        Array of points to compare against the mirror.
        Should have shape (npoint, 3).
    a : NDArray[np.floating]
        Coeffecients of the mirror function.
        Use a_primary for the primary mirror and a_secondary for the secondary.
    compensate : float, default: 0.0
        Amount to compensate the mirror surface by.
        This is useful to model things like the surface traced out by an SMR.

    Returns
    -------
    chisq : float
        The value to minimize when fitting to.
    """
    surface = mr.mirror(points[:, 0], points[:, 1], a, compensate)
    norm = mr.mirror_norm(points[:, 0], points[:, 1], a)
    res = (points[:, 2] - surface) * (norm[2] ** 2)

    return res @ res.T


def mirror_transform(
    transform_pars: NDArray[np.floating], points: NDArray[np.floating]
) -> NDArray[np.floating]:
    """
    Function to apply an affine transform to the mirror.
    This is the transform we are fitting for.

    Parameters
    ----------
    transform_pars : NDArray[np.floating]
        Flattened affine transform and shift, has to be 1d for use with minimizers.
        Should have shape (12,) where the first 9 elements are the flattened affine transform,
        and the last 3 are the shift in (x, y, z) applied after the affine transform.
    points : NDArray[np.floating]
        Array of points to compare against the mirror.
        Should have shape (npoint, 3).

    Returns
    -------
    points_transformed : NDArray[np.floating]
        Array of transformed points.
        Will have shape (npoint, 3).
    """
    aff = transform_pars[:9].reshape((3, 3))
    sft = transform_pars[9:]
    return points @ aff + sft


def mirror_fit(
    points: NDArray[np.floating],
    a: NDArray[np.floating],
    compensate: float = 0,
    to_points: bool = True,
    **kwargs,
) -> tuple[NDArray[np.floating], float]:
    """
    Fit points against the mirror surface.
    Ideally the points should be in the mirror's local coordinate system.

    Parameters
    ----------
    points : NDArray[np.floating]
        Array of points to compare against the mirror.
        Should have shape (npoint, 3).
    a : NDArray[np.floating]
        Coeffecients of the mirror function.
        Use a_primary for the primary mirror and a_secondary for the secondary.
    compensate : float, default: 0.0
        Amount to compensate the mirror surface by.
        This is useful to model things like the surface traced out by an SMR.
    to_points : bool, default: True
        If True, the transform will be inverted to align the model to the points.
    **kwargs
        Additional arguments to pass on to scipy.optimize.minimize.

    Returns
    -------
    transform_pars : NDArray[np.floating]
        Flattened affine transform and shift, has to be 1d for use with minimizers.
        Will have shape (12,) where the first 9 elements are the flattened affine transform,
        and the last 3 are the shift in (x, y, z) applied after the affine transform.
    rms : float
        The RMS error between the transformed points and the model.
    """

    def _fit_func(transform_pars, points, a, compensate):
        points_transformed = mirror_transform(transform_pars, points)
        chisq = mirror_objective(points_transformed, a, compensate)
        return chisq

    x0 = np.concatenate((np.eye(3).ravel(), np.zeros(3)))
    res = opt.minimize(_fit_func, x0, args=(points, a, compensate), **kwargs)

    transform_pars = res.x
    transformed = mirror_transform(transform_pars, points)
    z = mr.mirror(transformed[:, 0], transformed[:, 1], a, compensate)
    rms = np.sqrt(np.mean((z - transformed[:, 2]) ** 2))

    if to_points:
        aff = transform_pars[:9].reshape((3, 3))
        sft = transform_pars[9:]
        aff = np.linalg.inv(aff)
        sft = (-1 * sft) @ aff
        transform_pars = np.concatenate((aff.ravel(), sft))

    return transform_pars, rms


def tension_model(
    x0: float, y0: float, t: float, a: float, b: float, points: NDArray[np.floating]
) -> NDArray[np.floating]:
    """
    Function to model incorrect panel tensioning.
    Currently the model used is a radial power law.


    Parameters
    ----------
    x0 : float
        Center of the power law in x.
    y0 : float
        Center of the power law in y.
    t : float.
        Amplitude of power law,
        nominally the offset due to tensioning in the center of panel.
    a : float
        Base of power law.
    b : float
        Exponential scale factor of power law
    points : NDArray[np.floating]
        Points to compute power law at.
        Only the x and y coordinates are used (first two collumns).
        So should be (npoint, 2) but (npoint, ndim>2) is also fine.

    Returns
    -------
    z : NDArray[np.floating]
        Power law model at each xy.
        Will have shape (npoint,).
    """
    # Avoid divide by 0 error
    if a == 0:
        return np.zeros(len(points))

    # Compute radius at each point
    r = np.sqrt((points[:, 0] - x0) ** 2 + (points[:, 1] - y0) ** 2)

    # Return power law
    return t * (a ** (-b * r))


def tension_fit(
    residuals: NDArray[np.floating], **kwargs
) -> tuple[NDArray[np.floating], float]:
    """
    Fit a power law model of tension to a point cloud of residuals.

    Parameters
    ----------
    residuals : NDArray[np.floating]
        Residuals between measured point cloud and fit model.
    **kwargs
        Arguments to be passed to scipy.optimize.minimize

    Returns
    -------
    tension_pars : NDArray[np.floating]
        The fit parameters, see docstring of tension_model for details.
    rms : float
        The rms between the input residuals and the fit model.
    """

    def min_func(pars, residuals):
        _z = tension_model(*pars[:5], residuals)
        return np.sqrt(np.mean((residuals[:, 2] - _z) ** 2))

    if "bounds" not in kwargs:
        ptp = np.ptp(residuals[:, 2])
        bounds = [
            (np.min(residuals[:, 0]), np.max(residuals[:, 0])),
            (np.min(residuals[:, 1]), np.max(residuals[:, 1])),
            (-1 * ptp, ptp),
            (1e-10, np.inf),
            (0, np.inf),
        ]
        kwargs["bounds"] = bounds
    x0 = [np.mean(residuals[:, 0]), np.mean(residuals[:, 1]), 0, 1, 0]
    res = opt.minimize(min_func, x0, (residuals,), **kwargs)
    return res.x, res.fun


def res_auto_corr(
    residuals: NDArray[np.floating],
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Compute auto correlation of residuals from fit.

    Parameters
    ----------
    residuals : NDArray[np.floating]
        Residuals between measured point cloud and fit model.

    Returns
    -------
    ac : NDArray[np.floating]
        Auto correlation, really just the deviations in mm at each distance scale.
    ac_dists : NDArray[np.floating]
        Distance scale of each value in ac.
    """
    dists = np.zeros((len(residuals), len(residuals)))
    res_diff = np.zeros((len(residuals), len(residuals)))

    for i in range(len(residuals)):
        res1 = residuals[i]
        for j in range(i):
            res2 = residuals[j]
            dist = np.linalg.norm((res1[0] - res2[0], res1[1] - res2[1]))
            dists[i, j] = dist
            res_diff[i, j] = abs(res1[2] - res2[2])
    tri_i = np.tril_indices(len(residuals), k=-1)
    dists = dists[tri_i]
    res_diff = res_diff[tri_i]
    ac, bin_e, _ = binned_statistic(dists, res_diff, bins=100)
    ac_dists = bin_e[:-1] + np.diff(bin_e) / 2.0

    return ac, ac_dists
