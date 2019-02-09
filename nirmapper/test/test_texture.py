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
        self.texture = Texture(texture_path="tmp/fake_path/fake_texture.png", cam=self.cam)

    def test_check_occlusion_for_model(self):
        # downscale for preformat testing
        self.texture.cam.resolution_x = 40
        self.texture.cam.resolution_y = 20

        visible_triangle_ids = np.array([0, 1, 2, 4, 5, 11])
        calculated_visible_triangles = self.texture.check_occlusion_for_model(self.model)

        try:
            np.testing.assert_equal(calculated_visible_triangles, visible_triangle_ids)
            res = True
        except AssertionError as err:
            res = False
            print(err)
        self.assertTrue(res)

    def test_create_z_buffer(self):

        # downscale for preformat testing
        self.texture.cam.resolution_x = 40
        self.texture.cam.resolution_y = 20

        z_buffer = np.array([[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                             [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                             [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                             [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                             [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                             [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                             [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                             [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                             [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                             [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                             [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                             [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                             [-1, -1, -1, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 2, -1, -1, -1],
                             [-1, -1, -1, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 0, -1, -1, -1],
                             [-1, -1, -1, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 11, 0, -1, -1, -1],
                             [-1, -1, -1, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 11, 11, 0, -1, -1, -1],
                             [-1, -1, -1, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 11, 11, 11, 0, -1, -1, -1],
                             [-1, -1, -1, 5, 5, 5, 5, 5, 5, 5, 5, 5, 11, 11, 11, 11, 0, -1, -1, -1],
                             [-1, -1, -1, 5, 5, 5, 5, 5, 5, 5, 5, 11, 11, 11, 11, 11, 0, -1, -1, -1],
                             [-1, -1, -1, 5, 5, 5, 5, 5, 5, 5, 11, 11, 11, 11, 11, 11, 0, -1, -1, -1],
                             [-1, -1, -1, 5, 5, 5, 5, 5, 5, 5, 11, 11, 11, 11, 11, 11, 0, -1, -1, -1],
                             [-1, -1, -1, 5, 5, 5, 5, 5, 5, 11, 11, 11, 11, 11, 11, 11, 0, -1, -1, -1],
                             [-1, -1, -1, 5, 5, 5, 5, 5, 11, 11, 11, 11, 11, 11, 11, 11, 0, -1, -1, -1],
                             [-1, -1, -1, 5, 5, 5, 5, 11, 11, 11, 11, 11, 11, 11, 11, 11, 0, -1, -1, -1],
                             [-1, -1, -1, 1, 5, 5, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 0, -1, -1, -1],
                             [-1, -1, -1, 1, 5, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 0, -1, -1, -1],
                             [-1, -1, -1, 1, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 0, -1, -1, -1],
                             [-1, -1, -1, 11, 11, 11, -1, 11, 11, 11, 11, 4, 11, -1, 11, 4, 0, -1, -1, -1],
                             [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                             [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                             [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                             [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                             [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                             [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                             [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                             [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                             [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                             [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                             [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                             [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]])

        try:
            np.testing.assert_equal(self.texture.create_z_buffer(self.model)[:, :, 0], z_buffer)
            res = True
        except AssertionError as err:
            res = False
            print(err)
        self.assertTrue(res)

    def test_get_pixels_for_triangle(self):
        triangle = np.array([
            [1, 1, 1],
            [0.99, 1, 1],
            [1, 1, 0.99]
        ])

        expected = np.array([[610, 191],
                             [611, 191],
                             [612, 191],
                             [610, 192],
                             [611, 192],
                             [610, 193]])

        print(self.texture.get_pixels_for_triangle(triangle))

        try:
            np.testing.assert_equal(self.texture.get_pixels_for_triangle(triangle), expected)
            res = True
        except AssertionError as err:
            res = False
            print(err)
        self.assertTrue(res)

    def test_barycentric(self):
        triangle = np.array([
            [0, 0],
            [2, 0],
            [0, 2]
        ])

        p_inside1 = [1, 1]
        p_outside1 = [2, 0]
        p_outside2 = [2, 2]

        self.assertTrue(self.texture.barycentric(p_inside1, triangle))
        self.assertFalse(self.texture.barycentric(p_outside1, triangle))
        self.assertFalse(self.texture.barycentric(p_outside2, triangle))

        triangle2 = np.array([
            [12, 16],
            [27, 16],
            [27, 3]
        ])

        p_inside3 = [27, 3]

        self.assertTrue(self.texture.barycentric(p_inside3, triangle2))

    def test_get_bounding_box_coords_for_triangle(self):
        triangle = np.array([
            [1, 0],
            [2, 0],
            [1, 2]
        ])

        expected = np.array([
            [[1, 0], [2, 0]],
            [[1, 1], [2, 1]],
            [[1, 2], [2, 2]]
        ])

        try:
            np.testing.assert_equal(Texture.get_bounding_box_coords_for_triangle(triangle), expected)
            res = True
        except AssertionError as err:
            res = False
            print(err)
        self.assertTrue(res)

        triangle_2 = np.array([
            [0, 0],
            [2, 0],
            [0, 2]
        ])

        expected_2 = np.array([
            [[0, 0], [1, 0], [2, 0]],
            [[0, 1], [1, 1], [2, 1]],
            [[0, 2], [1, 2], [2, 2]]
        ])

        try:
            np.testing.assert_equal(Texture.get_bounding_box_coords_for_triangle(triangle_2), expected_2)
            res = True
        except AssertionError as err:
            res = False
            print(err)
        self.assertTrue(res)
