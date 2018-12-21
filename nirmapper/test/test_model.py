from unittest import TestCase
from unittest.mock import Mock, patch

import numpy as np

from nirmapper import Model
from nirmapper.model import IndicesFormat


class TestModel(TestCase):

    def setUp(self):
        self.model: Model = Model()
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
            0, 0, 0,
            1, 1, 1,
            2, 2, 2
        ])
        self.ind_format = "V3F_N3F_T2F"
        self.model.obj_vertices = self.vertices
        self.model.normals = self.normals
        self.model.uv_coords = self.uv_coords

    def test_obj_vertices(self):
        expected_vertices = np.array([
            [0.04, 0.05, 0.06],
            [0.01, 0.02, 0.03],
            [0.07, 0.08, 0.09]
        ])

        try:
            np.testing.assert_equal(self.model.obj_vertices, expected_vertices)
            res = True
        except AssertionError as err:
            res = False
            print(err)
        self.assertTrue(res)

        # Test setting empty ndarray

        self.model.obj_vertices = np.array([])

        try:
            np.testing.assert_equal(self.model.obj_vertices, np.array([]))
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

    def test_set_indices(self):
        expected_indices = np.array([
            [0, 0, 0],
            [1, 1, 1],
            [2, 2, 2]
        ])

        expected_format = [IndicesFormat.V3F, IndicesFormat.N3F, IndicesFormat.T2F]

        with patch.object(IndicesFormat, 'get_indices_formats_from_string') as mock_method:
            mock_method.return_value = [IndicesFormat.V3F, IndicesFormat.N3F, IndicesFormat.T2F]

        self.model.set_indices(self.indices, self.ind_format)

        try:
            np.testing.assert_equal(self.model.indices, expected_indices)
            res = True
        except AssertionError as err:
            res = False
            print(err)
        self.assertTrue(res)

        self.assertEqual(self.model.indices_format, expected_format)

        # Test setting empty ndarray

        self.model.set_indices(np.array([]), [])

        try:
            np.testing.assert_equal(self.model.indices, np.array([]))
            res = True
        except AssertionError as err:
            res = False
            print(err)
        self.assertTrue(res)

    def test_generate_indices(self):
        expected_indices = np.array([
            [0, 0, 0],
            [1, 1, 1],
            [2, 2, 2]
        ])

        expected_format = [IndicesFormat.V3F, IndicesFormat.N3F, IndicesFormat.T2F]

        indices, ind_format = self.model.generate_indices()

        try:
            np.testing.assert_equal(indices, expected_indices)
            res = True
        except AssertionError as err:
            res = False
            print(err)
        self.assertTrue(res)

        self.assertEqual(ind_format, expected_format)

        # Test less vertices

        self.model.normals = None

        expected_format = [IndicesFormat.V3F, IndicesFormat.T2F]

        expected_indices = np.array([
            [0, 0],
            [1, 1],
            [2, 2]
        ])

        indices, ind_format = self.model.generate_indices()

        try:
            np.testing.assert_equal(indices, expected_indices)
            res = True
        except AssertionError as err:
            res = False
            print(err)
        self.assertTrue(res)

        self.assertEqual(ind_format, expected_format)

        # Test empty vertices

        self.model.obj_vertices = None
        self.model.uv_coords = None
        indices, ind_format = self.model.generate_indices()

        try:
            np.testing.assert_equal(indices, np.array([]))
            res = True
        except AssertionError as err:
            res = False
            print(err)
        self.assertTrue(res)

        self.assertEqual(ind_format, [])

    def test_get_indices_for_format(self):
        indices = np.array([
            [0, 0, 0],
            [1, 1, 1],
            [2, 2, 2]
        ])

        formats = [IndicesFormat.V3F, IndicesFormat.N3F, IndicesFormat.T2F]

        with patch.object(IndicesFormat, 'get_indices_formats_from_string') as mock_method:
            mock_method.return_value = [IndicesFormat.V3F, IndicesFormat.N3F, IndicesFormat.T2F]

        self.model.set_indices(indices, formats)

        expected_indices = np.array([
            [0],
            [1],
            [2]
        ])

        ind_format = IndicesFormat.V3F

        try:
            np.testing.assert_equal(self.model.get_indices_for_format(ind_format), expected_indices)
            res = True
        except AssertionError as err:
            res = False
            print(err)
        self.assertTrue(res)

    def test_get_indices_for_not_existing_format(self):
        indices = np.array([
            [0, 0],
            [1, 1],
            [2, 2]
        ])

        formats = [IndicesFormat.V3F, IndicesFormat.N3F]

        with patch.object(IndicesFormat, 'get_indices_formats_from_string') as mock_method:
            mock_method.return_value = [IndicesFormat.V3F, IndicesFormat.N3F]

        self.model.set_indices(indices, formats)

        expected_indices = np.array([])

        ind_format = IndicesFormat.T2F

        try:
            np.testing.assert_equal(self.model.get_indices_for_format(ind_format), expected_indices)
            res = True
        except AssertionError as err:
            res = False
            print(err)
        self.assertTrue(res)

    def test_set_vertices_by_ind_format(self):
        self.model.normals = None

        normals = np.array([
            [0, 0, 0],
            [2, 2, 2],
            [3, 3, 3]
        ])

        ind_format = IndicesFormat.N3F

        self.model.set_vertices_by_ind_format(normals, ind_format)

        try:
            np.testing.assert_equal(self.model.normals, normals)
            res = True
        except AssertionError as err:
            res = False
            print(err)
        self.assertTrue(res)
