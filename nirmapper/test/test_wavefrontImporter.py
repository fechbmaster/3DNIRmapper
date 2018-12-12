import os
import numpy as np
from unittest import TestCase

from nirmapper.model import Wavefront


def prepend_dir(file):
    return os.path.join(os.path.dirname(__file__), file)


class TestWavefrontImporter(TestCase):

    def setUp(self):
        models = Wavefront.import_obj_as_model_list(prepend_dir('simple.obj'))
        self.model1 = models[0]
        self.model2 = models[1]

    def test_obj_vertices(self):

        model1_verts = np.array([
            0.04, 0.05, 0.06,
            0.01, 0.02, 0.03,
            0.07, 0.08, 0.09])

        model2_verts =  np.array([
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
