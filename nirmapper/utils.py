import math

import numpy as np

# epsilon for testing whether a number is close to zero
_EPS = np.finfo(float).eps * 4.0


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


def quaternion_matrix(quaternion):
    """Return homogeneous rotation matrix from quaternion.

    """
    q = np.array(quaternion, dtype=np.float64, copy=True)
    n = np.dot(q, q)
    if n < _EPS:
        return np.identity(4)
    q *= math.sqrt(2.0 / n)
    q = np.outer(q, q)
    return np.array([
        [1.0 - q[2, 2] - q[3, 3], q[1, 2] - q[3, 0], q[1, 3] + q[2, 0]],
        [q[1, 2] + q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] - q[1, 0]],
        [q[1, 3] - q[2, 0], q[2, 3] + q[1, 0], 1.0 - q[1, 1] - q[2, 2]]
    ])


def generate_triangle_sequence(vertices, indices) -> np.ndarray:
    """
    Method generates a triangle sequence out of vertices and their indices.
    It combines 3 vertices to a single row in the np.ndarray.
    :param vertices: The vertices.
    :param indices: The indices of the vertices.
    :return: Triangles np.ndarray with shape (x, 3, 3)
    """
    # Generate vertices sequence from describing indices
    vert_sequence = np.array(vertices[np.array(indices).flatten()])
    # Reshape the vert sequence to length/9x3x3 triangle Pairs
    triangles = vert_sequence.reshape(vert_sequence.size // 9, 3, 3)

    return triangles
