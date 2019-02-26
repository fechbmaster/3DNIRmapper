from unittest import TestCase

import numpy as np

from nirmapper.model import Model


class TestModel(TestCase):

    def setUp(self):
        self.model = Model()
        self.vertices = np.array([
            0.04, 0.05, 0.06,
            0.01, 0.02, 0.03,
            0.07, 0.08, 0.09
        ])
        self.normals = np.array([
            20.0, 21.0, 22.0,
            20.0, 21.0, 22.0,
            20.0, 21.0, 22.0,
        ])
        self.uv_coords = np.array([
            14.0, 15.0,
            12.0, 13.0,
            10.0, 11.0
        ])
        self.indices = np.array([
            0,
            1,
            2
        ])
        self.uv_indices = np.array([
            0,
            1,
            2
        ])
        self.normal_indices = np.array([
            0,
            1,
            2
        ])
        self.model.vertices = self.vertices
        self.model.indices = self.indices
        self.model.normals = self.normals
        self.model.normal_indices = self.normal_indices
        self.model.uv_coords = self.uv_coords
        self.model.uv_indices = self.uv_indices

    def test_obj_vertices(self):
        expected_vertices = np.array([
            [0.04, 0.05, 0.06],
            [0.01, 0.02, 0.03],
            [0.07, 0.08, 0.09]
        ])

        try:
            np.testing.assert_equal(self.model.vertices, expected_vertices)
            res = True
        except AssertionError as err:
            res = False
            print(err)
        self.assertTrue(res)

        # Test setting empty ndarray

        self.model.vertices = np.array([])

        try:
            np.testing.assert_equal(self.model.vertices, np.array([]))
            res = True
        except AssertionError as err:
            res = False
            print(err)
        self.assertTrue(res)

    def test_normals(self):
        expected_normals = np.array([
            [20.0, 21.0, 22.0],
            [20.0, 21.0, 22.0],
            [20.0, 21.0, 22.0],
        ])

        try:
            np.testing.assert_equal(self.model.normals, expected_normals)
            res = True
        except AssertionError as err:
            res = False
            print(err)
        self.assertTrue(res)

        # Test setting empty ndarray

        self.model.normals = np.array([])

        try:
            np.testing.assert_equal(self.model.normals, np.array([]))
            res = True
        except AssertionError as err:
            res = False
            print(err)
        self.assertTrue(res)

    def test_uv_coords(self):
        expected_uvs = np.array([
            [14.0, 15.0],
            [12.0, 13.0],
            [10.0, 11.0]
        ])

        try:
            np.testing.assert_equal(self.model.uv_coords, expected_uvs)
            res = True
        except AssertionError as err:
            res = False
            print(err)
        self.assertTrue(res)

        # Test setting empty ndarray

        self.model.uv_coords = np.array([])

        try:
            np.testing.assert_equal(self.model.uv_coords, np.array([]))
            res = True
        except AssertionError as err:
            res = False
            print(err)
        self.assertTrue(res)

    def test_indices(self):
        expected_indices = np.array([
            [0, 1, 2]
        ])

        try:
            np.testing.assert_equal(self.model.indices, expected_indices)
            np.testing.assert_equal(self.model.normal_indices, expected_indices)
            np.testing.assert_equal(self.model.uv_indices, expected_indices)
            res = True
        except AssertionError as err:
            res = False
            print(err)
        self.assertTrue(res)
