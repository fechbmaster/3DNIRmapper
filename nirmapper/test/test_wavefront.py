import os
import numpy as np
from unittest import TestCase

from nirmapper.model import Wavefront


def prepend_dir(file):
    return os.path.join(os.path.dirname(__file__), file)


class TestWavefront(TestCase):

    def setUp(self):
        self.wavefront = Wavefront(prepend_dir('resources/simple.obj'), cache=False)
        self.model1 = self.wavefront.models[0]
        self.model2 = self.wavefront.models[1]

    def test_obj_vertices(self):

        model1_verts = np.array([
            0.04, 0.05, 0.06,
            0.01, 0.02, 0.03,
            0.07, 0.08, 0.09])

        model2_verts = np.array([
            -1.0, 0.0, 1.0,
            1.0, 0.0, 1.0,
            1.0, 0.0, -1.0])

        try:
            np.testing.assert_almost_equal(self.model1.vertices, model1_verts)
            res = True
        except AssertionError as err:
            res = False
            print(err)
        self.assertTrue(res)

        try:
            np.testing.assert_almost_equal(self.model2.vertices, model2_verts)
            res = True
        except AssertionError as err:
            res = False
            print(err)
        self.assertTrue(res)

    def test_obj_normals(self):
        model1_norms = np.array([
            20.0, 21.0, 22.0,
            20.0, 21.0, 22.0,
            20.0, 21.0, 22.0,
        ])

        model2_norms = np.array([
            0.0, 1.0, -0.0,
            0.0, 1.0, -0.0,
            0.0, 1.0, -0.0
        ])

        try:
            np.testing.assert_almost_equal(self.model1.normals, model1_norms)
            res = True
        except AssertionError as err:
            res = False
            print(err)
        self.assertTrue(res)

        try:
            np.testing.assert_almost_equal(self.model2.normals, model2_norms)
            res = True
        except AssertionError as err:
            res = False
            print(err)
        self.assertTrue(res)

    def test_uv_coords(self):
        # uv_coords must not be imported - not necessary for this program
        self.assertEqual(len(self.model1.uv_coords), 0)
        self.assertEqual(len(self.model2.uv_coords), 0)

    def test_indices(self):
        indices = np.array([
            [0, 0],
            [1, 1],
            [2, 2],
            [3, 3],
            [4, 4],
            [5, 5],
            [6, 6],
            [7, 7],
            [8, 8]
        ])

        try:
            np.testing.assert_almost_equal(self.model1.indices, indices)
            res = True
        except AssertionError as err:
            res = False
            print(err)
        self.assertTrue(res)

        try:
            np.testing.assert_almost_equal(self.model2.indices, indices)
            res = True
        except AssertionError as err:
            res = False
            print(err)
        self.assertTrue(res)
