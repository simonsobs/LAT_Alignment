"""
Functions to describe the mirror surface.
"""
import numpy as np
from numpy.typing import NDArray

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


def mirror(
    x: float | NDArray[np.floating],
    y: float | NDArray[np.floating],
    a: NDArray[np.floating],
    compensate: float = 0.0,
) -> float | NDArray[np.floating]:
    """
    Analytic form for the mirror in the mirror's local coordinates.

    Paramaters
    ----------
    x : float|NDArray[np.floating]
        x positions to calculate the mirror at.
    y : float|NDArray[np.floating]
        y positions to calculate the mirror at.
    a : NDArray[np.floating]
        Coeffecients of the mirror function.
        Use a_primary for the primary mirror and a_secondary for the secondary.
    compensate : float, default: 0.0
        Amount to compensate the mirror surface by.
        This is useful to model things like the surface traced out by an SMR.

    Returns
    -------
    z : float|NDArray[np.floating]
        The surface of the mirror at the specified locations.
    """
    z = np.zeros_like(x)
    Rn = 3000.0
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            z += a[i, j] * (x / Rn) ** i * (y / Rn) ** j
    if compensate != 0.0:
        compensation = compensate * mirror_norm(x, y, a)
        z += compensation[:, 2]
    return z


def mirror_norm(
    x: float | NDArray[np.floating],
    y: float | NDArray[np.floating],
    a: NDArray[np.floating],
) -> NDArray[np.floating]:
    """
    Analytic form of mirror normal vector.

    Paramaters
    ----------
    x : float|NDArray[np.floating]
        x positions to calculate the mirror at.
    y : float|NDArray[np.floating]
        y positions to calculate the mirror at.
    a : NDArray[np.floating]
        Coeffecients of the mirror function.
        Use a_primary for the primary mirror and a_secondary for the secondary.
    compensate : float, default: 0.0
        Amount to compensate the mirror surface by.
        This is useful to model things like the surface traced out by an SMR.

    Returns
    -------
    normals : NDArray[np.floating]
        Unit vector normal to mirror at each xy.
        Has shape (npoints, 3).
    """
    Rn = 3000.0

    x_n = np.zeros_like(x)
    y_n = np.zeros_like(y)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            if i != 0:
                x_n += a[i, j] * (x ** (i - 1)) / (Rn**i) * (y / Rn) ** j
            if j != 0:
                y_n += a[i, j] * (x / Rn) ** i * (y ** (j - 1)) / (Rn**j)

    z_n = -1 * np.ones_like(x_n)
    normals = np.array((x_n, y_n, z_n)).T
    normals /= np.linalg.norm(normals, axis=-1)[:, np.newaxis]
    return normals
