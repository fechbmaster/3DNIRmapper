import numpy as np
import math
from collections import Iterable

from nirmapper.exceptions import ReshapeError


def euler_angles_to_rotation_matrix(theta):
    """
    Calculate a rotation matrix from given euler angle.

    :param theta: Euler angle in °
    :return: Rotation matrix
    """
    theta = np.radians(theta)

    R_x = np.array([[1, 0, 0],
                    [0, math.cos(theta[0]), -math.sin(theta[0])],
                    [0, math.sin(theta[0]), math.cos(theta[0])]
                    ])

    R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                    [0, 1, 0],
                    [-math.sin(theta[1]), 0, math.cos(theta[1])]
                    ])

    R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                    [math.sin(theta[2]), math.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])

    R = np.dot(R_z, np.dot(R_y, R_x))

    return R


def flatten(lis):
    """
    Function flattens a multidimensional list.

    :param lis: The list to flatten
    :return: Flattened list
    """
    for item in lis:
        if isinstance(item, Iterable) and not isinstance(item, str):
            for x in flatten(item):
                yield x
        else:
            yield item


def get_2d_coordinate_from_homogeneous_vector(vector):
    """
    Reverting homogeneous coordinates back to 2d coordinates

    :param vector: Homogeneous three-dimensional vector
    :return: Reverted two-dimensional coordinate
    """

    return vector[:-1]/vector[-1]
