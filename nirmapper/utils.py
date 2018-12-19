import math

import numpy as np


def euler_angles_to_rotation_matrix(theta):
    """
    Calculate a rotation matrix from given euler angle.

    :param theta: Euler angle in degrees.
    :return: Rotation matrix.
    """
    theta = np.radians(theta)

    cx = math.cos(theta[0])
    sx = math.sin(theta[0])
    cy = math.cos(theta[1])
    sy = math.sin(theta[1])
    cz = math.cos(theta[2])
    sz = math.sin(theta[2])

    R_x = np.array([[1, 0, 0],
                    [0, cx, -sx],
                    [0, sx, cx]
                    ])

    R_y = np.array([[cy, 0, sy],
                    [0, 1, 0],
                    [-sy, 0, cy]
                    ])

    R_z = np.array([[cz, -sz, 0],
                    [sz, cz, 0],
                    [0, 0, 1]
                    ])

    R = np.dot(R_z, np.dot(R_y, R_x))

    return R
