import numpy as np
from unittest import TestCase

from nirmapper.utils import euler_angles_to_rotation_matrix


class TestUtils(TestCase):

    def test_euler_angles_to_rotation_matrix(self):
        rotation = [90, 0, 0]

        expected = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0]])

        try:
            np.testing.assert_almost_equal(euler_angles_to_rotation_matrix(rotation), expected)
            res = True
        except AssertionError as err:
            res = False
            print(err)
        self.assertTrue(res)

        rotation_two = [-90, 180, 0]

        expected_two = np.array([[-1.0, 0.0, 0.0],
                                 [0.0, 0.0, 1.0],
                                 [-0.0, 1.0, -0.0]])

        try:
            np.testing.assert_almost_equal(euler_angles_to_rotation_matrix(rotation_two), expected_two)
            res = True
        except AssertionError as err:
            res = False
            print(err)
        self.assertTrue(res)

        rotation_three = [-90, 180, 45]

        expected_three = np.array([[-0.7071, -0.0, -0.7071],
                                   [-0.7071, -0.0, 0.7071],
                                   [0.0, 1.0, 0.0]])

        try:
            np.testing.assert_almost_equal(euler_angles_to_rotation_matrix(rotation_three).round(decimals=4),
                                           expected_three)
            res = True
        except AssertionError as err:
            res = False
            print(err)
        self.assertTrue(res)

