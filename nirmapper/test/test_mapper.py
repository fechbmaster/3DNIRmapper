import inspect
import numpy as np
import os
from unittest import TestCase

from nirmapper import Camera, Texture, Model
from nirmapper.mapper import Mapper


class TestMapper(TestCase):

    def setUp(self):
        scipt_path = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))
        texture_path = scipt_path + '/resources/images/texture_cube.png'
        output_path = '/tmp/cube_example.dae'
        print("This will create a demo mapping of a cube in ", output_path, " using the renderer from: ", texture_path)

        # Create Cam1

        location = [0, 7, 0]
        rotation = [-90, 180, 0]
        focal_length = 35
        sensor_width = 32
        sensor_height = 18
        screen_width = 1920
        screen_height = 1080

        cam1 = Camera(focal_length, screen_width, screen_height, sensor_width, sensor_height, location, rotation)

        # Create Cam2

        location = [7, 0, 0]
        rotation = [-90, 180, -90]
        focal_length = 35
        sensor_width = 32
        sensor_height = 18
        screen_width = 1920
        screen_height = 1080

        cam2 = Camera(focal_length, screen_width, screen_height, sensor_width, sensor_height, location, rotation)

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
        self.mapper = Mapper([texture1, texture2], model, 96, 54, "/tmp/test.dae", "TestCube")

    def test_visibility_analysis(self):
        exp_vis_ids1 = [5, 11]
        exp_vis_ids2 = [2, 8]

        self.mapper.start_visibility_analysis()

        try:
            np.testing.assert_equal(self.mapper.textures[0].vis_triangle_ids, exp_vis_ids1)
            np.testing.assert_equal(self.mapper.textures[1].vis_triangle_ids, exp_vis_ids2)
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
