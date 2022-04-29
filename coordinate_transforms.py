"""
Functions to transform between LAT coordinate systems.
See README for descriptions of each coordinate system.

Author: Saianeesh Keshav Haridas
"""
import numpy as np
import scipy.spatial as spat
from numpy import float64, ndarray
from typing import List, Tuple, Union

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
