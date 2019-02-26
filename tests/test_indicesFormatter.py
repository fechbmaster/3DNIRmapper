from unittest import TestCase

import numpy as np

from nirmapper.model import IndicesFormatter, IndicesFormat


class TestIndicesFormatter(TestCase):

    def setUp(self):
        self.all_verts = [14.0, 15.0, 20.0, 21.0, 22.0, 0.04, 0.05, 0.06, 12.0, 13.0, 20.0, 21.0, 22.0, 0.01, 0.02,
                          0.03,
                          10.0, 11.0, 20.0, 21.0, 22.0, 0.07, 0.08, 0.09]

        self.formats = [IndicesFormat.T2F, IndicesFormat.N3F, IndicesFormat.V3F]
        self.formatter = IndicesFormatter(self.formats)

    def test_get_uv_coords_by_format(self):
        expected_uv_coords = np.array([
            [14., 15.],
            [12., 13.],
            [10., 11.]
        ])

        try:
            np.testing.assert_equal(self.formatter.get_coords_by_format(self.all_verts, IndicesFormat.T2F),
                                    expected_uv_coords)
            res = True
        except AssertionError as err:
            res = False
            print(err)
        self.assertTrue(res)

    def test_get_normal_coords_by_format(self):
        expected_normal_coords = np.array([
            [20., 21., 22.],
            [20., 21., 22.],
            [20., 21., 22.]
        ])

        try:
            np.testing.assert_equal(self.formatter.get_coords_by_format(self.all_verts, IndicesFormat.N3F),
                                    expected_normal_coords)
            res = True
        except AssertionError as err:
            res = False
            print(err)
        self.assertTrue(res)

    def test_get_vertices_coords_by_format(self):
        expected_vertices_coords = np.array([
            [0.04, 0.05, 0.06],
            [0.01, 0.02, 0.03],
            [0.07, 0.08, 0.09]])

        try:
            np.testing.assert_equal(self.formatter.get_coords_by_format(self.all_verts, IndicesFormat.V3F),
                                    expected_vertices_coords)
            res = True
        except AssertionError as err:
            res = False
            print(err)
        self.assertTrue(res)

    def test_generate_indices(self):
        expected_indices = np.array([
            0, 1, 2, 3, 4, 5, 6, 7
        ])

        try:
            np.testing.assert_equal(self.formatter.generate_indices(8),
                                    expected_indices)
            res = True
        except AssertionError as err:
            res = False
            print(err)
        self.assertTrue(res)
