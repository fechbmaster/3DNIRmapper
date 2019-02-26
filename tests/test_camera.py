from unittest import TestCase

import numpy as np

from nirmapper.camera import Camera


class TestCamera(TestCase):

    def setUp(self):
        self.location = np.array([0, 7, 0])
        self.rotation = [-0.0, 0.0, 0.707, 0.707]
        self.focal_length = 35
        self.sensor_width = 32
        self.sensor_height = 18
        self.screen_width = 1920
        self.screen_height = 1080

        self.cam = Camera(self.focal_length, self.screen_width, self.screen_height, self.sensor_width,
                          self.sensor_height, self.location, self.rotation)

        self.p1 = np.array([1, 1, 1])
        self.p2 = np.array([1, 1, -1])
        self.p3 = np.array([-1, 1, -1])
        self.p4 = np.array([-1, 1, 1])

    def test_get_intrinsic_3x4_A_matrix(self):
        intrinsic_mat = np.array([
            [2100.0, 0.0, 960.0],
            [0.0, 2100.0, 540.0],
            [0.0, 0.0, 1.0]
        ])

        try:
            np.testing.assert_almost_equal(self.cam.get_intrinsic_3x4_A_matrix(), intrinsic_mat)
            res = True
        except AssertionError as err:
            res = False
            print(err)
        self.assertTrue(res)

    def test_get_extrinsic_3x4_D_matrix(self):
        extrinsic_mat = np.array([
            [-1.0000, 0.0000, 0.0000, 0.0000],
            [0.0000, 0.0000, -1.0000, 0.0000],
            [0.0000, -1.0000, 0.0000, 7.0000]
        ])

        try:
            np.testing.assert_almost_equal(self.cam.get_extrinsic_3x4_D_matrix(), extrinsic_mat)
            res = True
        except AssertionError as err:
            res = False
            print(err)
        self.assertTrue(res)

    def test_get_3x4_P_projection_matrix(self):
        projection_mat = np.array([
            [-2100.0, -960.0, 0.0, 6720.0],
            [0.0, -540.0, -2100.0, 3780.0],
            [0.0, -1.0, 0.0, 7.0]
        ])

        try:
            np.testing.assert_almost_equal(self.cam.get_3x4_P_projection_matrix(), projection_mat)
            res = True
        except AssertionError as err:
            res = False
            print(err)
        self.assertTrue(res)

    def test_project_world_point_to_pixel_coords(self):
        # test for single point...
        exp_p1 = np.array([610, 190])
        # ... and for an array
        exp_p2 = np.array([[610, 890],
                           [1310, 890],
                           [1310, 190]
                           ])

        pix_p1 = self.cam.get_pixel_coords_for_vertices(self.p1)
        pix_p2 = self.cam.get_pixel_coords_for_vertices(np.array([self.p2, self.p3, self.p4]))

        try:
            np.testing.assert_almost_equal(pix_p1, exp_p1)
            np.testing.assert_almost_equal(pix_p2, exp_p2)
            res = True
        except AssertionError as err:
            res = False
            print(err)
        self.assertTrue(res)

    def test_project_world_point_to_pixel_coords_including_z_value(self):
        exp_p1 = np.array([610, 190, 6])

        try:
            np.testing.assert_almost_equal(self.cam.get_pixel_coords_for_vertices(self.p1, include_z_value=True),
                                           exp_p1)
            res = True
        except AssertionError as err:
            res = False
            print(err)
        self.assertTrue(res)

    def test_project_world_points_to_uv_coords(self):
        # test for single point ...
        exp_p1 = np.array([610 / self.screen_width, (self.screen_height - 190) / self.screen_height])
        # ... and for array
        exp_p2 = np.array([610 / self.screen_width, (self.screen_height - 890) / self.screen_height])
        exp_p3 = np.array([1310 / self.screen_width, (self.screen_height - 890) / self.screen_height])
        exp_p4 = np.array([1310 / self.screen_width, (self.screen_height - 190) / self.screen_height])

        exp_array = np.array([exp_p2, exp_p3, exp_p4])

        pix_p1 = self.cam.get_texture_coords_for_vertices(self.p1)
        pix_p2 = self.cam.get_texture_coords_for_vertices(np.array([self.p2, self.p3, self.p4]))

        try:
            np.testing.assert_almost_equal(pix_p1, exp_p1)
            np.testing.assert_almost_equal(pix_p2, exp_array)
            res = True
        except AssertionError as err:
            res = False
            print(err)
        self.assertTrue(res)
