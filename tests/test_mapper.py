import inspect
import os
from unittest import TestCase

import numpy as np

from nirmapper.camera import Camera
from nirmapper.model import Texture, Model
from nirmapper.nirmapper import Mapper


class TestMapper(TestCase):

    def setUp(self):
        scipt_path = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))
        texture_path = scipt_path + '/resources/images/texture_cube.png'
        output_path = '/tmp/mapper_test.dae'

        # Create Cam1

        location = [0, 7, 0]
        rotation = [-90, 180, 0]
        focal_length = 35
        sensor_width = 32
        sensor_height = 18
        screen_width = 1920
        screen_height = 1080

        cam1 = Camera(focal_length, screen_width, screen_height, sensor_width, sensor_height, location, rotation,
                      "EULER")

        # Create Cam2

        location = [7, 0, 0]
        rotation = [-90, 180, -90]
        focal_length = 35
        sensor_width = 32
        sensor_height = 18
        screen_width = 1920
        screen_height = 1080

        cam2 = Camera(focal_length, screen_width, screen_height, sensor_width, sensor_height, location, rotation,
                      "EULER")

        # Create textures

        texture1 = Texture(texture_path, cam1)
        texture2 = Texture(texture_path, cam2)

        # Create model

        verts = np.array([[1, 1, -1],  # 1
                          [1, -1, -1],  # 2
                          [-1, -1, -1],  # 3
                          [-1, 1, -1],  # 4
                          [1, 1, 1],  # 5
                          [1, -1, 1],  # 6
                          [-1, -1, 1],  # 7
                          [-1, 1, 1]])  # 8

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

        indices = np.array([0,
                            2,
                            3,
                            7,
                            5,
                            4,
                            4,
                            1,
                            0,
                            5,
                            2,
                            1,
                            2,
                            7,
                            3,
                            0,
                            7,
                            4,
                            0,
                            1,
                            2,
                            7,
                            6,
                            5,
                            4,
                            5,
                            1,
                            5,
                            6,
                            2,
                            2,
                            6,
                            7,
                            0,
                            3,
                            7])

        normal_indices = indices

        model = Model(verts, normals)
        model.indices = indices
        model.normal_indices = normal_indices

        # Create Mapper
        self.mapper = Mapper([texture1, texture2], model, "/tmp/test.dae", "TestCube", 0.05)

    def test_visibility_analysis(self):
        exp_vis_ids1 = [5, 11]
        exp_vis_ids2 = [2, 8]

        self.mapper.start_visibility_analysis()

        try:
            np.testing.assert_equal(self.mapper.textures[0].vis_triangle_indices, exp_vis_ids1)
            np.testing.assert_equal(self.mapper.textures[1].vis_triangle_indices, exp_vis_ids2)
            res = True
        except AssertionError as err:
            res = False
            print(err)
        self.assertTrue(res)

        exp_vis_vertices1 = np.array([[1, 1, -1],
                                      [-1, 1, 1],
                                      [1, 1, 1],
                                      [1, 1, -1],
                                      [-1, 1, -1],
                                      [-1, 1, 1]])

        exp_vis_vertices2 = np.array([[1, 1, 1],
                                      [1, -1, -1],
                                      [1, 1, -1],
                                      [1, 1, 1],
                                      [1, -1, 1],
                                      [1, -1, -1]])

        try:
            np.testing.assert_equal(self.mapper.textures[0].visible_vertices, exp_vis_vertices1)
            np.testing.assert_equal(self.mapper.textures[1].visible_vertices, exp_vis_vertices2)
            res = True
        except AssertionError as err:
            res = False
            print(err)
        self.assertTrue(res)

    def test_clean_duplicates(self):
        # Create Cam3

        location = [4.28, 3.58, 0]
        rotation = [-90, 180, -52.2]
        focal_length = 35
        sensor_width = 32
        sensor_height = 18
        screen_width = 1920
        screen_height = 1080

        cam3 = Camera(focal_length, screen_width, screen_height, sensor_width, sensor_height, location, rotation,
                      "EULER")

        texture3 = Texture('/fake_texture.png', cam3)
        self.mapper.textures.append(texture3)
        self.mapper.start_visibility_analysis()
        self.mapper.clean_duplicates()

        expected_tri_ids_0 = [5, 11]
        expected_tri_ids_1 = [8]
        expected_tri_ids_2 = [2]

        try:
            np.testing.assert_equal(self.mapper.textures[0].vis_triangle_indices, expected_tri_ids_0)
            np.testing.assert_equal(self.mapper.textures[1].vis_triangle_indices, expected_tri_ids_1)
            np.testing.assert_equal(self.mapper.textures[2].vis_triangle_indices, expected_tri_ids_2)
            res = True
        except AssertionError as err:
            res = False
            print(err)
        self.assertTrue(res)

        try:
            np.testing.assert_equal(self.mapper.textures[0].duplicate_triangle_indices, [])
            np.testing.assert_equal(self.mapper.textures[1].duplicate_triangle_indices, [])
            np.testing.assert_equal(self.mapper.textures[2].duplicate_triangle_indices, [])
            res = True
        except AssertionError as err:
            res = False
            print(err)
        self.assertTrue(res)

    def test_set_duplicates_for_textures(self):
        # Create Cam3

        location = [4.28, 3.58, 0]
        rotation = [-90, 180, -52.2]
        focal_length = 35
        sensor_width = 32
        sensor_height = 18
        screen_width = 1920
        screen_height = 1080

        cam3 = Camera(focal_length, screen_width, screen_height, sensor_width, sensor_height, location, rotation,
                      "EULER")

        texture3 = Texture('/fake_texture.png', cam3)
        self.mapper.textures.append(texture3)

        self.mapper.start_visibility_analysis()
        self.mapper.set_duplicates_for_textures()

        expected_dups_0 = [5, 11]
        expected_dups_1 = [2, 8]
        expected_dups_2 = [2, 5, 8, 11]

        try:
            np.testing.assert_equal(self.mapper.textures[0].duplicate_triangle_indices, expected_dups_0)
            np.testing.assert_equal(self.mapper.textures[1].duplicate_triangle_indices, expected_dups_1)
            np.testing.assert_equal(self.mapper.textures[2].duplicate_triangle_indices, expected_dups_2)
            res = True
        except AssertionError as err:
            res = False
            print(err)
        self.assertTrue(res)

    def test_get_best_texture_for_id(self):
        # Create Cam3

        location = [4.28, 3.58, 0]
        rotation = [-90, 180, -52.2]
        focal_length = 35
        sensor_width = 32
        sensor_height = 18
        screen_width = 1920
        screen_height = 1080

        cam3 = Camera(focal_length, screen_width, screen_height, sensor_width, sensor_height, location, rotation,
                      "EULER")

        texture3 = Texture('/fake_texture.png', cam3)
        self.mapper.textures.append(texture3)

        self.mapper.start_visibility_analysis()
        self.mapper.set_duplicates_for_textures()

        self.assertEqual(self.mapper.get_best_texture_for_duplicate_triangle(2), 2,
                         "Best Texture for triangle 2 is at index 2")
        self.assertEqual(self.mapper.get_best_texture_for_duplicate_triangle(8), 1,
                         "Best Texture for triangle 8 is at index 1")
        self.assertEqual(self.mapper.get_best_texture_for_duplicate_triangle(5), 0,
                         "Best Texture for triangle 5 is at index 0")
        self.assertEqual(self.mapper.get_best_texture_for_duplicate_triangle(11), 0,
                         "Best Texture for triangle 11 is at index 0")
