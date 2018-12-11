import os
from unittest import TestCase

from nirmapper.model import WavefrontImporter


def prepend_dir(file):
    return os.path.join(os.path.dirname(__file__), file)


class TestWavefrontImporter(TestCase):

    def setUp(self):
        models = WavefrontImporter.import_obj_as_model_list(prepend_dir('simple.obj'))
        self.model1 = models[0]
        self.model2 = models[1]

    def test_import_obj_vertices(self):
        self.assertEqual(self.model1.vertices, [
            0.01, 0.02, 0.03,
            0.04, 0.05, 0.06,
            0.07, 0.08, 0.09,
            0.11, 0.12, 0.13])

        self.assertEqual(self.model2.vertices, [
            1.0, 0.0, 1.0,
            -1.0, 0.0, 1.0,
            1.0, 0.0, -1.0,
            -1.0, 0.0, -1.0])




