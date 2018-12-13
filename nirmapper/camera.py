import numpy as np

from .utils import (
    euler_angles_to_rotation_matrix
)


class Camera(object):
    """Camera class.

    The Camera class holds the intrinsic and extrinsic camera parameters.
    """

    def __init__(self,
                 focal_length_in_mm,
                 resolution_x, resolution_y,
                 sensor_width_in_mm, sensor_height_in_mm,
                 cam_location_xyz, cam_euler_rotation_theta):
        self.focal_length_in_mm = focal_length_in_mm
        self.resolution_x = resolution_x
        self.resolution_y = resolution_y
        self.sensor_width_in_mm = sensor_width_in_mm
        self.sensor_height_in_mm = sensor_height_in_mm
        self.cam_location_xyz = cam_location_xyz
        self.cam_euler_rotation_theta = cam_euler_rotation_theta

    def get_intrinsic_3x4_A_matrix(self) -> np.ndarray:
        """
        Get intrinsic camera calibration matrix K.

        :return: Calibration Matrix K
        """
        s_u = self.resolution_x / self.sensor_width_in_mm
        s_v = self.resolution_y / self.sensor_height_in_mm

        alpha_u = self.focal_length_in_mm * s_u
        alpha_v = self.focal_length_in_mm * s_v
        u_0 = self.resolution_x / 2
        v_0 = self.resolution_y / 2
        skew = 0  # use only rectangular pixels

        A = np.array([[alpha_u, skew, u_0],
                      [0, alpha_v, v_0],
                      [0, 0, 1]
                      ])

        return A

    def get_extrinsic_3x4_D_matrix(self) -> np.ndarray:
        """
        Get extrinsic camera matrix consisting of rotation and transformation matrix.

        :return: Extrinsic matrix RT
        """
        R_cam2cv = np.array([[1, 0, 0],
                             [0, -1, 0],
                             [0, 0, -1]
                             ])

        R_world2cam = euler_angles_to_rotation_matrix(self.cam_euler_rotation_theta).T
        T_world2cam = np.dot(-1 * R_world2cam, self.cam_location_xyz)

        R_world2cv = np.dot(R_cam2cv, R_world2cam)
        T_world2cv = R_cam2cv.dot(T_world2cam).reshape(3, 1)

        RT = np.hstack((R_world2cv, T_world2cv))

        return RT

    def get_3x4_P_projection_matrix(self) -> np.ndarray:
        """
        Get combined camera matrix consisting of intrinsic (A) and extrinsic (Rt) matrix.

        :return: Combined matrix P
        """
        A = self.get_intrinsic_3x4_A_matrix()
        D = self.get_extrinsic_3x4_D_matrix()

        return np.dot(A, D)
