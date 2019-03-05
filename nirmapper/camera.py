from enum import Enum
from typing import Union, List

import numpy as np

from nirmapper.utils import (
    quaternion_matrix, euler_angles_to_rotation_matrix)


class RotationFormat(Enum):
    EULER = 0
    QUAT = 1


class Camera(object):
    """Camera class.

    The Camera class holds the intrinsic and extrinsic camera parameters.
    """

    def __init__(self,
                 focal_length_in_mm: float,
                 resolution_x: int, resolution_y: int,
                 sensor_width_in_mm: float, sensor_height_in_mm: float,
                 cam_location_xyz: Union[List[float], np.ndarray],
                 rotation: Union[List[float], np.ndarray],
                 rotation_type: Union[RotationFormat, str] = RotationFormat.QUAT):
        self.focal_length_in_mm = focal_length_in_mm
        self.resolution_x = resolution_x
        self.resolution_y = resolution_y
        self.sensor_width_in_mm = sensor_width_in_mm
        self.sensor_height_in_mm = sensor_height_in_mm
        self.cam_location_xyz = cam_location_xyz
        self.rotation = rotation
        if isinstance(rotation_type, str):
            self.rotation_type = RotationFormat[rotation_type]
        else:
            self.rotation_type = rotation_type

        if self.rotation_type is RotationFormat.EULER:
            if np.size(rotation) != 3:
                raise ValueError("Wrong shape of rotation coordinates for euler rotation.")
        elif self.rotation_type is RotationFormat.QUAT:
            if np.size(rotation) != 4:
                raise ValueError("Wrong shape of rotation coordinates for quaternion rotation.")

        self.A = np.array([])
        self.D = np.array([])
        self.P = np.array([])

    def get_intrinsic_3x4_A_matrix(self) -> np.ndarray:
        """
        Get intrinsic camera calibration matrix K.

        :return: Calibration Matrix K
        """
        if self.A.size == 0:
            s_u = self.resolution_x / self.sensor_width_in_mm
            s_v = self.resolution_y / self.sensor_height_in_mm

            alpha_u = self.focal_length_in_mm * s_u
            alpha_v = self.focal_length_in_mm * s_v
            u_0 = self.resolution_x / 2
            v_0 = self.resolution_y / 2
            skew = 0  # use only rectangular pixels

            A = np.array([
                [alpha_u, skew, u_0],
                [0, alpha_v, v_0],
                [0, 0, 1]
            ])
            self.A = A

        return self.A

    def get_extrinsic_3x4_D_matrix(self) -> np.ndarray:
        """
        Get extrinsic camera matrix consisting of rotation and transformation matrix.

        :return: Extrinsic matrix RT
        """
        if self.D.size == 0:
            R_cam2cv = np.array([[1, 0, 0],
                                 [0, -1, 0],
                                 [0, 0, -1]
                                 ])

            # Transpose since the rotation is object rotation, and coordinate rotation is needed
            if self.rotation_type is RotationFormat.QUAT:
                R_world2cam = quaternion_matrix(self.rotation).T
            elif self.rotation_type is RotationFormat.EULER:
                R_world2cam = euler_angles_to_rotation_matrix(self.rotation).T
            else:
                raise ValueError("Rotation Format is not supported!")
            T_world2cam = np.dot(-1 * R_world2cam, self.cam_location_xyz)

            R_world2cv = np.dot(R_cam2cv, R_world2cam)
            T_world2cv = R_cam2cv.dot(T_world2cam).reshape(3, 1)

            RT = np.hstack((R_world2cv, T_world2cv))

            self.D = RT

        return self.D

    def get_3x4_P_projection_matrix(self) -> np.ndarray:
        """
        Get combined camera matrix consisting of intrinsic (A) and extrinsic (Rt) matrix.

        :return: Combined matrix P
        """
        if self.P.size == 0:
            A = self.get_intrinsic_3x4_A_matrix()
            D = self.get_extrinsic_3x4_D_matrix()
            P = np.dot(A, D)

            self.P = P

        return self.P

    def get_pixel_coords_for_vertices(self, points: np.ndarray, include_z_value: bool = False) -> np.ndarray:
        """
        Calculates the image space coordinates for coordinates in object space. The coordinate values will
        be between the image pixel height and width.

        :param numpy.array points: The three-dimensional coordinates in object space
        :param bool include_z_value: Indicates if z_values should be included in return value or not
        :return numpy.array: Two-dimensional pixel coordinates in image space
        """
        # User maybe just passed a single coord - convert to 2d array
        dim = points.ndim
        if dim == 1:
            points = np.array([points])

        pixel_coords = []
        for point in points:
            P = self.get_3x4_P_projection_matrix()
            # Append 3d coord with homogeneous coord to calculate coordinate
            uv_xyz = P.dot(np.append(point, 1))
            # Normalize by dividing trough third component
            uv_xy = np.array(uv_xyz[:-1] / uv_xyz[-1])
            if include_z_value:
                pixel_coords.append(np.append(uv_xy, uv_xyz[-1:]))
            else:
                pixel_coords.append(uv_xy)

        pixel_coords = np.around(np.array(pixel_coords)).astype(int)

        # Convert back to single entry if user passed a single point
        if dim == 1:
            pixel_coords = pixel_coords.flatten()

        return pixel_coords

    def get_texture_coords_for_vertices(self, points: np.ndarray) -> np.ndarray:
        """
        Calculates the image space UV coordinates for coordinates in object space.
        UV coordinates are between 0 and 1.

        :param numpy.array points: The three-dimensional coordinates in object space
        :return numpy.array: Two-dimensional uv coordinates in image space
        """
        # User maybe just passed a single coord - convert to 2d array
        dim = points.ndim
        if dim == 1:
            points = np.array([points])

        uv_coords = []
        for point in points:
            uv_xy = self.get_pixel_coords_for_vertices(point)
            uv_coords.append(uv_xy)

        uv_coords = np.array(uv_coords, dtype=float)

        uv_coords[:, [0]] = uv_coords[:, [0]] / self.resolution_x
        uv_coords[:, [1]] = (self.resolution_y - uv_coords[:, [1]]) / self.resolution_y

        if dim == 1:
            uv_coords = uv_coords.flatten()

        return np.array(uv_coords)
