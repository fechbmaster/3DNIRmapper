import numpy as np
from .utils import (
    get_2d_coordinate_from_homogeneous_vector
)
from nirmapper.camera import Camera


class UVMapper(object):
    """UVMapper class.

    The UVMapper class is responsible for mapping coordinates from 3D-model to uv coordinates.

    """

    @staticmethod
    def calculate_image_space_pixel_coordinate(object_space_coord, cam:  Camera):
        """
        Calculates the image space coordinates for a coordinate in object space. The coordinate values will
        be between the image pixel height and width.

        :param numpy.array object_space_coord: The three-dimensional coordinates in object space
        :param Camera cam: The camera
        :return numpy.array: Two-dimensional pixel coordinates in image space
        """
        P = cam.get_3x4_P_projection_matrix()
        # Append 3d coord with fourth param to calculate coordinate
        uv_xyz = P.dot(np.append(object_space_coord, 1))
        uv_xy = get_2d_coordinate_from_homogeneous_vector(uv_xyz)

        return uv_xy

    @staticmethod
    def calculate_uv_coordinate(object_space_coord, cam: Camera):
        """
        Calculates the image space UV coordinates for a coordinate in object space. UV coordinates are between 0 and 1.

        :param numpy.array object_space_coord: The three-dimensional coordinates in object space
        :param Camera cam: The camera
        :return numpy.array: Two-dimensional uv coordinates in image space
        """

        uv_xy = UVMapper.calculate_image_space_pixel_coordinate(object_space_coord, cam)
        uv_xy[0] = uv_xy[0] / cam.resolution_x
        uv_xy[1] = uv_xy[1] / cam.resolution_y

        return uv_xy

    @staticmethod
    def calculate_uv_coordinates(object_space_coords, cam: Camera):
        """
        Calculate the UV coordinates for a bunch of coords.  UV coordinates are between 0 and 1.

        :param numpy.array object_space_coords: The
        :param Camera cam: The camera
        :return numpy.array: List of
        """
        uv_coords = []
        for coord in object_space_coords:
            uv_coords.append(UVMapper.calculate_uv_coordinate(coord, cam))

        return np.array(uv_coords).flatten()
