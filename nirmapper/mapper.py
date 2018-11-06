import numpy as np
from .utils import (
    get_2d_coordinate_from_homogeneous_vector
)


class UVMapper(object):

    @staticmethod
    def calculate_uv_coordinates_for_vector(vector, cam):
        P = cam.get_3x4_P_matrix(cam)
        uv_xyz = P.dot(vector)
        uv_xy = get_2d_coordinate_from_homogeneous_vector(uv_xyz)

        return uv_xy

    @staticmethod
    def calculate_uv_coordinates_for_vertex(vertex, cam):
        P = cam.get_3x4_P_matrix(cam)
        uv_xyz = P.dot(np.append(vertex, 1))
        uv_xy = get_2d_coordinate_from_homogeneous_vector(uv_xyz)

        return uv_xy
