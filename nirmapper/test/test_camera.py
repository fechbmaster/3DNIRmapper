import numpy as np
from unittest import TestCase

from nirmapper import Camera


class TestCamera(TestCase):

    def setUp(self):
        self.location = np.array([0, 7, 0])
        self.rotation = np.array([-90, 0, 0])
        self.focal_length = 35
        self.sensor_width = 32
        self.sensor_height = 18
        self.screen_width = 1920
        self.screen_height = 1080

        self.cam = Camera(self.focal_length, self.screen_width, self.screen_height, self.sensor_width, self.sensor_height, self.location, self.rotation)

    def test_get_intrinsic_3x4_A_matrix(self):
        intrinsic_mat = np.array([
            [2100.0,    0.0,    960.0],
            [0.0,       2100.0, 540.0],
            [0.0,       0.0,    1.0]
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
            [1.0000, 0.0000, 0.0000, -0.0000],
            [0.0000, 0.0000, 1.0000, -0.0000],
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
            [2100.0, -960.0, 0.0, 6720.0],
            [0.0, -540.0, 2100.0, 3780.0],
            [0.0, -1.0, 0.0, 7.0]
        ])

        try:
            np.testing.assert_almost_equal(self.cam.get_3x4_P_projection_matrix(), projection_mat)
            res = True
        except AssertionError as err:
            res = False
            print(err)
        self.assertTrue(res)
