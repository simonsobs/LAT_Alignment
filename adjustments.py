import numpy as np
from scipy.spatial.transform import Rotation as rot

def rotate(point, end_point1, end_point2, thetha):
    origin = np.mean(end_point1, end_point2)
    point_0 = point - origin
    ax = end_point2 - end_point1
    ax = rot.from_rotvec(thetha * ax/np.linalg.norm(ax))
    point_0 = ax.apply(point_0)
    return point_0 + origin


def rotate_panel(points, adjustors, thetha_0, thetha_1):
    rot_points = np.zeros(points.shape)
    rot_adjustors = np.zeros(adjustors.shape)
    for i in range(len(points)):
        rot_points[i] = rotate(points[i], adjustors[1], adjustors[2], thetha_0)
        rot_adjustors[i] = rotate(adjustors[i], adjustors[1], adjustors[2], thetha_0)
    for i in range(len(points)):
        rot_points[i] = rotate(rot_points[i], adjustors[0], adjustors[3], thetha_1)
        rot_adjustors[i] = rotate(rot_adjustors[i], adjustors[0], adjustors[3], thetha_1)
    return rot_points, rot_adjustors


def translate_panel(points, adjustors, dx, dy):
    translation = np.array((dx, dy, 0))
    return points + translation, adjustors + translation
