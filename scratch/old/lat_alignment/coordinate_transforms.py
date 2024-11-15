"""
Functions to transform between LAT coordinate systems.
See README for descriptions of each coordinate system.

Author: Saianeesh Keshav Haridas
"""

from typing import List, Optional, Tuple, Union

import megham.transform as mt
import numpy as np
import scipy.spatial as spat
from numpy import float64, ndarray
from numpy.typing import NDArray

import lat_alignment.fitting as lf
import lat_alignment.mirror as mr

v_m1 = (0, 0, 3600)  # mm
v_m2 = (0, -4800, 0)  # mm
v_c = (-200, 0, 0)
a_m1 = -np.arctan(0.5)
a_m2 = np.arctan(1.0 / 3.0) - np.pi / 2


def global_to_cad(coords, shift):
    """
    Transform from global coordinates to cad coordinates

    @param coords: Array of points to transform
    @param shift: Origin shift in mm

    @return m_coords:
    """
    shifted_coords = coords - shift
    m_coords = np.zeros(shifted_coords.shape)
    m_coords[:, 0] = shifted_coords[:, 1]
    m_coords[:, 1] = 1 * shifted_coords[:, 0]
    m_coords[:, 2] = -1 * shifted_coords[:, 2]
    return m_coords + v_c


def cad_to_global(coords: ndarray, shift: Union[List[float], int]) -> ndarray:
    """
    Transform from cad coordinates to global coordinates

    @param coords: Array of points to transform
    @param shift: Origin shift in mm

    @return m_coords:
    """
    shifted_coords = coords - shift - v_c
    m_coords = np.zeros(shifted_coords.shape)
    m_coords[:, 0] = 1 * shifted_coords[:, 1]
    m_coords[:, 1] = shifted_coords[:, 0]
    m_coords[:, 2] = -1 * shifted_coords[:, 2]
    return m_coords


def global_to_mirror(
    coords: ndarray, shift: int, v_m: Tuple[int, int, int], a_m: float64
) -> ndarray:
    """
    Transform from global coordinates to mirror coordinates

    @param coords: Array of points to transform
    @param shift: Origin shift in mm (in addition to the standard one for the mirror)
    @param v_m: Standard origin shift for the mirror
    @param a_m: Angle for coordinate rotation (reffered to as alpha in Vertex docs)

    @return m_coords: The points in the mirror coords
    """
    shifted_coords = coords - shift - v_m
    rot_mat = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(a_m), np.sin(a_m)],
            [0.0, -np.sin(a_m), np.cos(a_m)],
        ]
    )
    m_coords = np.zeros(shifted_coords.shape)
    for i, point in enumerate(shifted_coords):
        m_coords[i] = rot_mat @ point
    return m_coords


def global_to_primary(coords, shift):
    """
    Transform from global coordinates to primary mirror coordinates

    @param coords: Array of points to transform
    @param shift: Origin shift in mm (in addition to the standard one for the mirror)

    @return m_coords: The points in the primary mirror coords
    """
    return global_to_mirror(coords, shift, v_m1, a_m1)


def global_to_secondary(coords: ndarray, shift: int) -> ndarray:
    """
    Transform from global coordinates to secondary mirror coordinates

    @param coords: Array of points to transform
    @param shift: Origin shift in mm (in addition to the standard one for the mirror)

    @return m_coords: The points in the secondary mirror coords
    """
    return global_to_mirror(coords, shift, v_m2, a_m2)


def mirror_to_global(coords, shift, v_m, a_m):
    """
    Transform from mirror coordinates to global coordinates

    @param coords: Array of points to transform
    @param shift: Origin shift in mm (in addition to the standard one for the mirror)
    @param v_m: Standard origin shift for the mirror
    @param a_m: Angle for coordinate rotation (reffered to as alpha in Vertex docs)

    @return m_coords: The points in global coords
    """
    shifted_coords = coords - shift
    rot_mat = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(a_m), -np.sin(a_m)],
            [0.0, np.sin(a_m), np.cos(a_m)],
        ]
    )
    m_coords = np.zeros(shifted_coords.shape)
    for i, point in enumerate(shifted_coords):
        m_coords[i] = rot_mat @ point
    return m_coords + v_m


def primary_to_global(coords, shift):
    """
    Transform from primary mirror coordinates to global coordinates

    @param coords: Array of points to transform
    @param shift: Origin shift in mm (in addition to the standard one for the mirror)

    @return m_coords: The points in global mirror coords
    """
    return mirror_to_global(coords, shift, v_m1, a_m1)


def secondary_to_global(coords, shift):
    """
    Transform from secondary mirror coordinates to global coordinates

    @param coords: Array of points to transform
    @param shift: Origin shift in mm (in addition to the standard one for the mirror)

    @return m_coords: The points in global mirror coords
    """
    return mirror_to_global(coords, shift, v_m2, a_m2)


def primary_to_secondary(coords, shift):
    """
    Transform from primary mirror coordinates to secondary mirror coordinates

    @param coords: Array of points to transform
    @param shift: Origin shift in mm (in addition to the standard one for the mirror)

    @return m_coords: The points in secondary mirror coords
    """
    global_coords = primary_to_global(coords, shift)
    return global_to_secondary(global_coords, 0)


def secondary_to_primary(coords, shift):
    """
    Transform from secondary mirror coordinates to primary mirror coordinates

    @param coords: Array of points to transform
    @param shift: Origin shift in mm (in addition to the standard one for the mirror)

    @return m_coords: The points in primary mirror coords
    """
    global_coords = secondary_to_global(coords, shift)
    return global_to_primary(global_coords, 0)


def cad_to_primary(coords, shift):
    """
    Transform from cad coordinates to primary mirror coordinates

    @param coords: Array of points to transform
    @param shift: Origin shift in mm (in addition to the standard one for the mirror)

    @return m_coords: The points in primary mirror coords
    """
    global_coords = cad_to_global(coords, shift)
    return global_to_primary(global_coords, 0)


def cad_to_secondary(coords: ndarray, shift: Union[List[float], int]) -> ndarray:
    """
    Transform from cad coordinates to secondary mirror coordinates

    @param coords: Array of points to transform
    @param shift: Origin shift in mm (in addition to the standard one for the mirror)

    @return m_coords: The points in secondary mirror coords
    """
    global_coords = cad_to_global(coords, shift)
    return global_to_secondary(global_coords, 0)


def primary_to_cad(coords, shift):
    """
    Transform from primary mirror coordinates to cad coordinates

    @param coords: Array of points to transform
    @param shift: Origin shift in mm (in addition to the standard one for the mirror)

    @return m_coords: The points in cad mirror coords
    """
    global_coords = primary_to_global(coords, shift)
    return global_to_cad(global_coords, 0)


def secondary_to_cad(coords, shift):
    """
    Transform from primary mirror coordinates to cad coordinates

    @param coords: Array of points to transform
    @param shift: Origin shift in mm (in addition to the standard one for the mirror)

    @return m_coords: The points in cad mirror coords
    """
    global_coords = secondary_to_global(coords, shift)
    return global_to_cad(global_coords, 0)


def shift_coords(coords, shift):
    """
    Apply origin shift to coordinate

    @param coords: Array of points to transform
    @param shift: Origin shift in mm (in addition to the standard one for the mirror)

    @return m_coords: The points in the secondary mirror coords
    """
    return coords - shift


def compensate(coords, compensation):
    """
    Copensate measurement from FARO by applying a shift normal to the surface

    @param coords: Array of points to transform
    @param compensation: Compensation in mm

    @return c_coords: The compensated coordinates
    """
    # Complute Delaunay tessellation of point cloud
    simplices = spat.Delaunay(coords, True).simplices

    # Initialize array of normal vectors
    norms = np.zeros(coords.shape)
    for sim in simplices:
        # Since the input is 3D, each simplex is a tetrahedron
        # Calculate unit normal vector at each vertex
        for i in range(4):
            vect_1 = coords[sim[(i + 1) % 4]] - coords[sim[i]]
            vect_2 = coords[sim[(i - 1) % 4]] - coords[sim[i]]
            # vect_3 = coords[sim[(i + 2) % 4]] - coords[sim[i]]
            norm_vec = np.cross(vect_1, vect_2)
            # flip = np.sign((np.dot(norm_vec, vect_3)))
            norm_vec /= np.linalg.norm(norm_vec)
            # norms[sim[i]] += flip * norm_vec
            norms[sim[i]] += norm_vec

    # Get average unit normal vector at each point
    norms /= np.linalg.norm(norms, axis=1)[:, np.newaxis]

    return coords - compensation * norms


def reference_align(
    points: NDArray[np.floating],
    source_names: NDArray[np.str_],
    source_points: NDArray[np.str_],
    target_names: NDArray[np.str_],
    target_points: NDArray[np.floating],
) -> NDArray[np.floating]:
    """
    Get an affine transform based on reference points.
    Useful when you have a measurement in some arbitrary coordinates
    and want to align them to known ones.

    The names of the points are used to set the registration.

    Parameters
    ----------
    points: NDArray[np.floating]
        The points to take from the source to target coordinates.
        Should have shape (ndim, ndim).
    source_names : NDArray[np.str_]
        The names of the source points.
        Should be in the same order as source_points.
        Should have shape (nsource,)
    source_points : NDArray[np.floating]
        The reference points in the source coordinates.
        Should have shape (nsource, ndim).
    target_names : NDArray[np.str_]
        The names of the target points.
        Should be in the same order as target_points.
        Should have shape (ntarget,)
    target_points : NDArray[np.floating]
        The reference points in the target coordinates.
        Should have shape (ntarget, ndim).

    Returns
    -------
    transformed : NDArray[np.floating]
        The points in the target coordinates.
        Has shape (npoint, ndim).
    """
    _, idx_s, idx_t = np.intersect1d(source_names, target_names, return_indices=True)
    aff, sft = mt.get_affine(source_points[idx_s], target_points[idx_t])

    return points @ aff + sft


def surface_align(
    points: NDArray[np.floating],
    a: NDArray[np.floating],
    compensate: float = 0,
    bounds: Optional[list[tuple[float, float]]] = None,
    niter: int = 3,
) -> NDArray[np.floating]:
    """
    Align points to the mirror surface.
    This is useful if you think there is some systematic in your measurement
    or you don't trust your alignment.

    Paramaters
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
    bounds: Optional[list[tuple[float, float]]], default: None
        Bounds on the fit.
        If None some reasonible defaults are used.
    niter : int, default: 3
        Number of iteration to do.
        Between each iteration points that are a poor fit are thrown out.

    Returns
    -------
    transformed : NDArray[np.floating]
        The points in the target coordinates.
        Has shape (ngoodpoint, ndim), where ngoodpoint <= npoint.
    """
    if bounds is None:
        bounds = [(0.0, 2.0)] * 9 + [(-50.0, 50.0)] * 3
    for i in niter:
        t_pars, _ = lf.mirror_fit(points, a, compensate, bounds=bounds, to_points=False)
        t_points = lf.mirror_transform(t_pars, points)
        ares = np.abs(
            mr.mirror(t_points[:, 0], t_points[:, 1], a, compensate) - t_points[:, 2]
        )
        msk = ares < np.median(ares) + 3 * np.std(ares)
        points = t_points[msk]
    return t_points
