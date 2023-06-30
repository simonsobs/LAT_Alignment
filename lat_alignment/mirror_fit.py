"""
Fit point cloud to the analytic functional form of the LAT mirror surfaces

Author: Saianeesh Keshav Haridas
"""
import numpy as np
import scipy.optimize as opt
from scipy.spatial.transform import Rotation as rot
from scipy.stats import binned_statistic
from numpy import float64, ndarray
from typing import Callable, Tuple, Union

# fmt: off
a_primary = np.array([
            [0., 0., -57.74022, 1.5373825, 1.154294, -0.441762, 0.0906601,],
            [0., 0., 0., 0., 0., 0., 0.],
            [-72.17349, 1.8691899, 2.8859421, -1.026471, 0.2610568, 0., 0.],
            [0., 0., 0., 0., 0., 0., 0.],
            [1.8083973, -0.603195, 0.2177414, 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0.],
            [0.0394559, 0., 0., 0., 0., 0., 0.,]
            ])

a_secondary = np.array([
            [0., 0., 103.90461, 6.6513025, 2.8405781, -0.7819705, -0.0400483,  0.0896645],
            [0., 0., 0., 0., 0., 0., 0., 0.],
            [115.44758, 7.3024355, 5.7640389, -1.578144, -0.0354326, 0.2781226, 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0.],
            [2.9130983, -0.8104051, -0.0185283, 0.2626023, 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0.],
            [-0.0250794, 0.0709672, 0., 0., 0., 0., 0., 0.,],
            [0., 0., 0., 0., 0., 0., 0., 0.]
            ])
# fmt: on


def mirror(x: ndarray, y: ndarray, a: ndarray) -> ndarray:
    """
    Analytic form for the mirror

    @param x: x positions to calculate at
    @param y: y positions to calculate at
    @param a: Coeffecients of the mirror function
              Use a_primary for the primary mirror
              Use a_secondary for the secondary mirror

    @return z: z position of the mirror at each xy
    """
    z = np.zeros_like(x)
    Rn = 3000.0
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            z += a[i, j] * (x / Rn) ** i * (y / Rn) ** j
    return z


def mirror_norm(x: ndarray, y: ndarray, a: ndarray) -> ndarray:
    """
    Analytic form of mirror normal vector

    @param x: x positions to calculate at
    @param y: y positions to calculate at
    @param a: Coeffecients of the mirror function
              Use a_primary for the primary mirror
              Use a_secondary for the secondary mirror

    @return normals: Unit vector normal to mirror at each xy
    """
    Rn = 3000.0

    x_n = np.zeros_like(x)
    y_n = np.zeros_like(y)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            if i != 0:
                x_n += a[i, j] * (x ** (i - 1)) / (Rn ** i) * (y / Rn) ** j
            if j != 0:
                y_n += a[i, j] * (x / Rn) ** i * (y ** (j - 1)) / (Rn ** j)

    z_n = -1 * np.ones_like(x_n)
    normals = np.array((x_n, y_n, z_n)).T
    normals /= np.linalg.norm(normals, axis=-1)[:, np.newaxis]
    return normals


def mirror_fit_func(
    xy: Tuple[ndarray, ndarray],
    compensate: float,
    x0: float64,
    y0: float64,
    z0: float64,
    a1: float64,
    a2: float64,
    a3: float64,
    a: ndarray,
) -> ndarray:
    """
    Function to fit against for primary mirror
    Note that since each panel adjusts independantly it is reccomended to fit on a per panel basis

    @param xy: Tuple where first element is array of x points and second is y
    @param compensate: Amount to compensate for Faro measurement
                       Should be the radius of the SMR
    @param x0: x offset
    @param y0: y offset
    @param z0: z offset
    @param a1: Rotation about x axis
    @param a2: Rotation about y axis
    @param a3: Rotation about z axis
    @param a: Coeffecients of the mirror function
              Use a_primary for the primary mirror
              Use a_secondary for the secondary mirror

    @return z: The z position of the mirror at each xy
    """
    x = xy[0] - x0
    y = xy[1] - y0
    z = mirror(x, y, a) - z0

    if compensate != 0.0:
        compensation = compensate * mirror_norm(x, y, a)
        x += compensation[:, 0]
        y += compensation[:, 1]
        z += compensation[:, 2]

    xyz = np.zeros(x.shape + (3,))
    xyz[:, 0] = x
    xyz[:, 1] = y
    xyz[:, 2] = z

    ax1 = rot.from_rotvec(a1 * np.array([1.0, 0.0, 0.0]))
    ax2 = rot.from_rotvec(a2 * np.array([0.0, 1.0, 0.0]))
    ax3 = rot.from_rotvec(a3 * np.array([0.0, 0.0, 1.0]))
    ax = ax1 * ax2 * ax3
    xyz = ax.apply(xyz)

    return xyz[:, 2]


def primary_fit_func(xy, compensate, x0, y0, z0, a1, a2, a3):
    """
    Function to fit against for primary mirror
    Note that since each panel adjusts independantly it is reccomended to fit on a per panel basis

    @param xy: Tuple where first element is array of x points and second is y
    @param compensate: Amount to compensate for Faro measurement
                       Should be the radius of the SMR
    @param x0: x offset
    @param y0: y offset
    @param z0: z offset
    @param a1: Rotation about x axis
    @param a2: Rotation about y axis
    @param a3: Rotation about z axis

    @return z: The z position of the mirror at each xy
    """
    return mirror_fit_func(xy, compensate, x0, y0, z0, a1, a2, a3, a_primary)


def secondary_fit_func(
    xy: Tuple[ndarray, ndarray],
    compensate: float,
    x0: float64,
    y0: float64,
    z0: float64,
    a1: float64,
    a2: float64,
    a3: float64,
) -> ndarray:
    """
    Function to fit against for secondary mirror
    Note that since each panel adjusts independantly it is reccomended to fit on a per panel basis

    @param xy: Tuple where first element is array of x points and second is y
    @param compensate: Amount to compensate for Faro measurement
                       Should be the radius of the SMR
    @param x0: x offset
    @param y0: y offset
    @param z0: z offset
    @param a1: Rotation about x axis
    @param a2: Rotation about y axis
    @param a3: Rotation about z axis

    @return z: The z position of the mirror at each xy
    """
    return mirror_fit_func(xy, -1 * compensate, x0, y0, z0, a1, a2, a3, a_secondary)


def mirror_fit(
    x: ndarray, y: ndarray, z: ndarray, compensate: float, fit_func: Callable, **kwargs
) -> Tuple[ndarray, float64]:
    """
    Fit a cloud of points to mirror surface
    Note that since each panel adjusts independantly it is reccomended to fit on a per panel basis

    @param x: x position of each point
    @param y: y position of each point
    @param z: z position of each point
    @param compensate: Amount to compensate for Faro measurement
                       Should be the radius of the SMR
    @param fit_func: Function to fit against
                     For primary use primary_fit_func
                     For secondary use secondary_fit_func
    @param **kwargs: Arguments to be passed to scipy.optimize.minimize

    @return popt: The fit parameters, see docstring of the fit_func for details
    @return rms: The rms between the measured points and the fit model
    """

    def min_func(pars, x, y, z, compensate):
        _z = fit_func((x, y), compensate, *pars)
        return np.sqrt(np.mean((z - _z) ** 2))

    res = opt.minimize(min_func, np.zeros(6), (x, y, z, compensate), **kwargs)
    return res.x, res.fun


def tension_fit_func(
    residuals: ndarray, x0: float64, y0: float64, t: float64, a: float64, b: float64
) -> Union[int, ndarray]:
    """
    Function to fit for incorrect panel tensioning from residuals
    Currently the model used is a radial power law

    @param residuals: Residuals between measured point cloud and fit model
                      Nominally generated with calc_residuals
    @param x0: Offset to center mean subtracted x coordinates
    @param y0: Offset to center mean subtracted y coordinates
    @param t: Difference due to tensioning in the center of panel
    @param a: Base of power law
    @param b: Exponential scale factor of power law

    @return z: Power law model at each xy
    """
    # Avoid divide by 0 error
    if a == 0:
        return 0

    # Get average x and y position
    x_cm = residuals[:, 0].mean()
    y_cm = residuals[:, 1].mean()

    # Compute radius at each point
    r = np.sqrt((residuals[:, 0] - x_cm - x0) ** 2 + (residuals[:, 1] - y_cm - y0) ** 2)

    # Return power law
    return t * (a ** (-b * r))


def tension_fit(residuals: ndarray, **kwargs) -> Tuple[ndarray, float64]:
    """
    Fit a power law model of tension to a point cloud of residuals

    @param residuals: Residuals between measured point cloud and fit model
                      Nominally generated with calc_residuals
    @param **kwargs: Arguments to be passed to scipy.optimize.minimize

    @return popt: The fit parameters, see docstring of tension_fit_func for details
    @return rms: The rms between the measured points and the fit model
    """

    def min_func(pars, residuals):
        _z = tension_fit_func(residuals, *pars)
        return np.sqrt(np.mean((residuals[:, 2] - _z) ** 2))

    res = opt.minimize(min_func, np.zeros(5), (residuals,), **kwargs)
    return res.x, res.fun


def transform_point(
    points: ndarray,
    x0: float64,
    y0: float64,
    z0: float64,
    a1: float64,
    a2: float64,
    a3: float64,
) -> ndarray:
    """
    Transform points from model space to real (measured) space

    @param points: Points to compute at, either a 1d or 2d array
                   Each point should be (x, y, z)
    @param x0: x offset
    @param y0: y offset
    @param z0: z offset
    @param a1: Rotation about x axis
    @param a2: Rotation about y axis
    @param a3: Rotation about z axis

    @return points: The positions of the points in real space
    """
    ndims = len(points.shape)
    if ndims == 1:
        points = np.array([points])

    real_points = points - np.array([x0, y0, z0])

    ax1 = rot.from_rotvec(a1 * np.array([1.0, 0.0, 0.0]))
    ax2 = rot.from_rotvec(a2 * np.array([0.0, 1.0, 0.0]))
    ax3 = rot.from_rotvec(a3 * np.array([0.0, 0.0, 1.0]))
    ax = ax1 * ax2 * ax3
    rot_points = ax.apply(real_points)

    real_points[:, 2] = rot_points[:, 2]

    if ndims == 1:
        return real_points[0]
    return real_points


def get_delta(points, x0, y0, z0, a1, a2, a3):
    """
    Get the xyz offset of a point in real space from model space
    Nominally used to compute how far off the adjustment points are

    @param points: Points to compute at, either a 1d or 2d array
                   Each point should be (x, y, z)
    @param x0: x offset
    @param y0: y offset
    @param z0: z offset
    @param a1: Rotation about x axis
    @param a2: Rotation about y axis
    @param a3: Rotation about z axis

    @return delta: The xyz offsets of each point, same shape as points
                   Currently model - real
    """
    real_points = transform_point(points, x0, y0, z0, a1, a2, a3)
    return points - real_points


def calc_residuals(
    x: ndarray,
    y: ndarray,
    z: ndarray,
    compensate: float,
    fit_func: Callable,
    x0: float64,
    y0: float64,
    z0: float64,
    a1: float64,
    a2: float64,
    a3: float64,
) -> ndarray:
    """
    Calculate residuals from fit

    @param x: x position of each point
    @param y: y position of each point
    @param z: z position of each point
    @param compensate: Amount to compensate for Faro measurement
                       Should be the radius of the SMR
    @param fit_func: Function to fit against
                     For primary use primary_fit_func
                     For secondary use secondary_fit_func
    @param x0: x offset
    @param y0: y offset
    @param z0: z offset
    @param a1: Rotation about x axis
    @param a2: Rotation about y axis
    @param a3: Rotation about z axis

    @return residuals: The residuals
    """
    _z = fit_func((x, y), compensate, x0, y0, z0, a1, a2, a3)
    return np.array((x, y, z - _z)).T


def res_power_spect(residuals):
    """
    Compute power spectrum of residuals from fit
    Note that this is technically not the PSD,
    to convert to the PSD just divide by the range of distances in the output

    @param residuals: Residuals to compute power spectrum from

    @return ps: Power spectrum, really just the deviations in mm at each distance scale
    @return ps_dists: Distance scale of each value in ps
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
    ps, bin_e, bin_n = binned_statistic(dists, res_diff, bins=100)
    ps_dists = bin_e[:-1] + np.diff(bin_e) / 2.0

    return ps, ps_dists
