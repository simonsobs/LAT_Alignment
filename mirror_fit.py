"""
Fit point cloud to the analytic functional form of the LAT mirror surfaces

Author: Saianeesh Keshav Haridas
"""
import numpy as np
import scipy.optimize as opt
from scipy.spatial.transform import Rotation as rot

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


def mirror(x, y, a):
    """
    Analyitic form for the mirror

    @param x: x positions to calculate at
    @param y: y positions to calculate at
    @param a: Coeffecients of the mirror function
              Use a_primary for the primary mirror
              Use a_secondary for the secondary mirror

    @return z: z position of the mirror at each xy
    """
    z = 0.0
    Rn = 3000.0
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            z += a[i, j] * (x / Rn) ** i * (y / Rn) ** j
    return z


def mirror_fit_func(xy, x0, y0, z0, a1, a2, a3, a):
    """
    Function to fit against for primary mirror
    Note that since each panel adjusts independantly it is reccomended to fit on a per panel basis

    @param xy: Tuple where first element is array of x points and second is y
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

    xyz = np.zeros(x.shape + (3,))
    xyz[:, 0] = x
    xyz[:, 1] = y
    xyz[:, 2] = z

    origin = np.array((x.max() + x.min(), y.max() + y.min(), z.max() + z.min())) / 2.0
    xyz -= origin

    ax1 = rot.from_rotvec(a1 * np.array([1., 0., 0.]))
    ax2 = rot.from_rotvec(a2 * np.array([0., 1., 0.]))
    ax3 = rot.from_rotvec(a3 * np.array([0., 0., 1.]))
    ax = ax1 * ax2 * ax3
    xyz = ax.apply(xyz) + origin

    return xyz[:, 2]


def primary_fit_func(xy, x0, y0, z0, a1, a2, a3):
    """
    Function to fit against for primary mirror
    Note that since each panel adjusts independantly it is reccomended to fit on a per panel basis

    @param xy: Tuple where first element is array of x points and second is y
    @param x0: x offset
    @param y0: y offset
    @param z0: z offset
    @param a1: Rotation about x axis
    @param a2: Rotation about y axis
    @param a3: Rotation about z axis

    @return z: The z position of the mirror at each xy
    """
    return mirror_fit_func(xy, x0, y0, z0, a1, a2, a3, a_primary)


def secondary_fit_func(xy, x0, y0, z0, a1, a2, a3):
    """
    Function to fit against for secondary mirror
    Note that since each panel adjusts independantly it is reccomended to fit on a per panel basis

    @param xy: Tuple where first element is array of x points and second is y
    @param x0: x offset
    @param y0: y offset
    @param z0: z offset
    @param a1: Rotation about x axis
    @param a2: Rotation about y axis
    @param a3: Rotation about z axis

    @return z: The z position of the mirror at each xy
    """
    return mirror_fit_func(xy, x0, y0, z0, a1, a2, a3, a_secondary)


def mirror_fit(x, y, z, fit_func, **kwargs):
    """
    Fit a cloud of points to mirror surface
    Note that since each panel adjusts independantly it is reccomended to fit on a per panel basis

    @param x: x position of each point
    @param y: y position of each point
    @param z: z position of each point
    @param fit_func: Function to fit against
                     For primary use primary_fit_func
                     For secondary use secondary_fit_func
    @param **kwargs: Arguments to be passed to scipy.optimize.curve_fit

    @return popt: The fit parameters, see docstring of the fit_func for details
    @return rms: The rms between the measured points and the fit model
    """
    # popt, pcov = opt.curve_fit(fit_func, (x, y), z, **kwargs)
    # z_fit = fit_func((x, y), *popt)
    def min_func(pars, x, y, z):
        _z = fit_func((x, y), *pars)
        return np.sqrt(np.mean((z - _z) ** 2))

    res = opt.minimize(min_func, np.zeros(6), (x, y, z))
    return res.x, res.fun


def transform_point(points, x0, y0, z0, a1, a2, a3):
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
    
    x = real_points[:, 0]
    y = real_points[:, 1]
    z = real_points[:, 2]
    origin = np.array((x.max() + x.min(), y.max() + y.min(), z.max() + z.min())) / 2.0
    real_points -= origin

    ax1 = rot.from_rotvec(a1 * np.array([1., 0., 0.]))
    ax2 = rot.from_rotvec(a2 * np.array([0., 1., 0.]))
    ax3 = rot.from_rotvec(a3 * np.array([0., 0., 1.]))
    ax = ax1 * ax2 * ax3
    real_points = ax.apply(real_points) + origin

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
