import numpy as np

from .utils import (
    euler_angles_to_rotation_matrix,
    get_2d_coordinate_from_homogeneous_vector)


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

            A = np.array([[alpha_u, skew, u_0],
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

            R_world2cam = euler_angles_to_rotation_matrix(self.cam_euler_rotation_theta).T
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

    def project_world_point_to_pixel_coords(self, point: np.ndarray):
        """
        Calculates the image space coordinates for a coordinate in object space. The coordinate values will
        be between the image pixel height and width.

        :param numpy.array point: The three-dimensional coordinates in object space
        :return numpy.array: Two-dimensional pixel coordinates in image space
        """
        P = self.get_3x4_P_projection_matrix()
        # Append 3d coord with fourth param to calculate coordinate
        uv_xyz = P.dot(np.append(point, 1))
        uv_xy = get_2d_coordinate_from_homogeneous_vector(uv_xyz)

        return uv_xy

    def project_world_point_to_uv_coords(self, point: np.ndarray):
        """
        Calculates the image space UV coordinates for a coordinate in object space.
        UV coordinates are between 0 and 1.

        :param numpy.array point: The three-dimensional coordinates in object space
        :return numpy.array: Two-dimensional uv coordinate in image space
        """
        uv_xy = self.project_world_point_to_pixel_coords(point)
        uv_xy[0] = uv_xy[0] / self.resolution_x
        uv_xy[1] = uv_xy[1] / self.resolution_y

        return np.array(uv_xy)

    def project_world_points_to_uv_coords(self, points: np.ndarray):
        """
        Calculates the image space UV coordinates for coordinates in object space.
        UV coordinates are between 0 and 1.

        :param numpy.array points: The three-dimensional coordinates in object space
        :return numpy.array: Two-dimensional uv coordinates in image space
        """
        # User maybe just passed a single coord - fallback to other function
        if points.ndim == 1:
            return self.project_world_point_to_uv_coords(points)

        uv_coords = []
        for point in points:
            uv_xy = self.project_world_point_to_uv_coords(point)
            uv_coords.append(uv_xy)

        return np.array(uv_coords)
