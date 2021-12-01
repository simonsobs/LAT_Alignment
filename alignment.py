import adjustments as adj
import mirror_fit as mf
import numpy as np

"""
File structure:
    Each measurement session will have its own directory with some useful naming scheme
        Something like YYYYMMDD_num is probably good
    Each mirror will have its own subdir
    Within each mirror directory have a file for each panel whose name is the panel name
    File will contain pointcloud of measurements
    In root of directory we need some sort of config file that tells you what coordinate system the measurements were taken in as well as any adjustments that need to be applied to the model (ie: an origin shift to account for something that is in the wrong place but can't be moved)
        (these could also just be command line arguments)
    Also need to have some sort of lookup table that contains the positions of the alignmnt points and adjustors for each panel (in the mirror coordinates)
    
Workflow:
    Read in config file/parse command line arguments
    Load measurements on a per panel basis
    Transform panel from measurement coordinates to mirror coordinates
    Fit using mf
    Transform adjustor and alignment point locations with fit params
    Fit for adjustments with adj
    Print out adjustments and save to a file in root of measurement dir
"""

v_m1 = (0, 0, 3600) #mm
v_m2 = (0, -4800, 0) #mm
a_m1 = -np.arctan(0.5)
a_m2 = np.arctan(1./3.) - np.pi/2

def global_to_mirror(coords, shift, v_m, a_m):
    """
    Transform from global coordinates to mirror coordinates

    @param coords: Array of points to transform
    @param shift: Origin shift in mm (in addition to the standard one for the mirror)
    @param v_m: Standard origin shift for the mirror
    @param a_m: Angle for coordinate rotation (reffered to as alpha in Vertex docs)

    @return m_coords: The points in the mirror coords
    """
    shifted_coords = coords - shift - v_m
    rot_mat = np.array([[1., 0., 0.],
                        [0., np.cos(a_m), np.sin(a_m)],
                        [0., -np.sin(a_m), np.cos(a_m)]])
    m_coords = np.zeros(shifted_coords.shape)
    for i,point in enumerate(shifted_coords):
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


def global_to_secondary(coords, shift):
    """
    Transform from global coordinates to secondary mirror coordinates

    @param coords: Array of points to transform
    @param shift: Origin shift in mm (in addition to the standard one for the mirror)

    @return m_coords: The points in the secondary mirror coords
    """
    return global_to_mirror(coords, shift, v_m2, a_m2)
