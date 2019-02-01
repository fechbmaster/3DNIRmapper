from unittest import TestCase

import numpy as np

from nirmapper import Model, Camera
from nirmapper.texture import Texture


class TestTexture(TestCase):

    def setUp(self):
        location = np.array([0, 7, 0])
        rotation = np.array([-90, 180, 0])
        focal_length = 35
        sensor_width = 32
        sensor_height = 18
        screen_width = 1920
        screen_height = 1080

        self.cam = Camera(focal_length, screen_width, screen_height, sensor_width, sensor_height, location, rotation)

        verts = np.array([[1, 1, -1],  # 0
                          [1, -1, -1],  # 1
                          [-1, -1, -1],  # 2
                          [-1, 1, -1],  # 3
                          [1, 1, 1],  # 4
                          [1, -1, 1],  # 5
                          [-1, -1, 1],  # 6
                          [-1, 1, 1]])  # 7

        normals = np.array([0, 0, -1,
                            0, 0, 1,
                            1, 0, 0,
                            0, -1, 0,
                            -1, 0, 0,
                            0, 1, 0,
                            0, 0, -1,
                            0, 0, 1,
                            1, 0, 0,
                            0, -1, 0,
                            -1, 0, 0,
                            0, 1, 0])

        indices = np.array([0, 0,
                            2, 2,
                            3, 3,
                            7, 7,
                            5, 5,
                            4, 4,
                            4, 4,
                            1, 1,
                            0, 0,
                            5, 5,
                            2, 2,
                            1, 1,
                            2, 2,
                            7, 7,
                            3, 3,
                            0, 0,
                            7, 7,
                            4, 4,
                            0, 0,
                            1, 1,
                            2, 2,
                            7, 7,
                            6, 6,
                            5, 5,
                            4, 4,
                            5, 5,
                            1, 1,
                            5, 5,
                            6, 6,
                            2, 2,
                            2, 2,
                            6, 6,
                            7, 7,
                            0, 0,
                            3, 3,
                            7, 7])

        model = Model(vertices=verts, normals=normals)
        model.set_indices(indices, ind_format="V3F_N3F")
        self.model = model
        self.texture = Texture(text_id=1, texture_path="tmp/fake_path/fake_texture.png", cam=self.cam)


    def test_check_occlusion_for_model(self):
        visible_verts = [
            [-1,  1, 1],
            [-1,  1, -1],
            [1,   1, -1],
            [1,   1,  1]
        ]

        self.model.get_triangles()

        calculated_visible_verts= self.texture.check_occlusion_for_model(self.model)

        self.assertEqual(visible_verts, calculated_visible_verts)
