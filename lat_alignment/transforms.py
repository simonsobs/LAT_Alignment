from typing import Optional
import numpy as np
from numpy.typing import NDArray
from scipy.spatial import transform
from megham.transform import get_affine, apply_transform
from megham.utils import make_edm
import scipy.spatial as spat
from numpy import float64, ndarray
from typing import List, Tuple, Union

DEFAULT_REF = {"primary": [], "secondary": [((np.nan, np.nan, np.nan), ["CODE41", "CODE42", "CODE43"]),((np.nan, np.nan, np.nan), ["CODE32", "CODE33", "CODE34"]),((np.nan, np.nan, np.nan), ["CODE35", "CODE36", "CODE37"]),((np.nan, np.nan, np.nan), ["CODE38", "CODE39", "CODE40"])]}


v_m1 = (0, 0, 3600)  # mm
v_m2 = (0, -4800, 0)  # mm
v_c = (-200, 0, 0)
a_m1 = -np.arctan(0.5)
a_m2 = np.arctan(1.0 / 3.0) - np.pi / 2

def align_photo(labels: NDArray[np.str_], coords: NDArray[np.float32], *, mirror: str="primary", reference: Optional[list[tuple[tuple[float, float, float], list[str]]]] =None, shift: tuple[float, float, float]=(0, 0, 0), max_dist: float=100.) -> NDArray[np.float32]:
    """
    Align photogrammetry data and then put it into mirror coordinates.
    
    Parameters
    ----------
    labels : NDArray[np.str_]
        The labels of each photogrammetry point.
        Should have shape `(npoint,)`.
    coords : NDArray[np.float32]
        The coordinates of each photogrammetry point.
        Should have shape `(npoint, 3)`.
    mirror : str, default: 'primary'
        The mirror that these points belong to.
        Should be either: 'primary' or 'secondary'.
    reference : Optional[list[tuple[tuple[float, float, float], list[str]]]], default: None
        List of reference points to use.
        Each point given should be a tuple with two elements.
        The first element is a tuple with the (x, y, z) coordinates
        of the point in the global coordinate system.
        The second is a list of nearby coded targets that can be used
        to identify the point.
        If `None` the default reference for each mirror is used.
    shift : tuple[float, float, float], default: (0, 0, 0)
        Shift to apply before transforming into mirror coordinates.
    max_dist : float, default: 100
        Max distance in mm that the reference poing can be from the target
        point used to locate it.

    Returns
    -------
    coords_transformed : NDArray[np.float32]
        The transformed coordinates.
    """
    if mirror not in ["primary", "secondary"]:
        raise ValueError(f"Invalid mirror: {mirror}")
    if mirror == "primary":
        transform = global_to_primary
    else:
        transform = global_to_secondary
    if reference is None:
        reference = DEFAULT_REF[mirror]
    if reference is None or len(reference) == 0:
        raise ValueError("Invalid or empty reference")

    # Lets find the points we can use
    trg_idx = np.where(np.char.find(labels, "TARGET") > 0)[0]
    ref = []
    pts = []
    for rpoint, codes in reference:
        have = np.isin(codes, labels)
        if np.sum(have) == 0:
            continue
        coded = coords[np.where(labels == codes[np.where(have)[0][0]])[0][0]]
        # Find the closest point
        dist = np.linalg.norm(coords[trg_idx] - coded, axis=-1)
        if np.min(dist) > max_dist:
            continue
        ref += [rpoint]
        pts += [coords[trg_idx][np.argmin(dist)]]

    if len(ref) < 4:
        raise ValueError(f"Only {len(ref)} reference points found! Can't align!")

    aff, sft = get_affine(np.vstack(pts), np.vstack(ref))
    coords_transformed = apply_transform(coords, aff, sft)

    coords_transformed = transform(coords_transformed, np.array(shift))
    return coords_transformed


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
