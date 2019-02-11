from unittest import TestCase

import numpy as np

from nirmapper import Camera, Model
from nirmapper.model.model import IndicesFormat
from nirmapper.renderer.renderer import Renderer


class TestRenderer(TestCase):

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

        self.renderer = Renderer()

    def test_get_visible_triangles(self):
        # downscale for preformat testing
        buffer_resolution_x = 40
        buffer_resolution_y = 20

        visible_triangle_ids = np.array([5, 11])
        visible_triangle_counts = np.array([99, 96])
        vis_triangles, counts = \
            self.renderer.get_visible_triangles(self.model.vertices,
                                                self.model.get_indices_for_format(IndicesFormat.V3F),
                                                self.cam,
                                                buffer_resolution_x, buffer_resolution_y)

        try:
            np.testing.assert_equal(vis_triangles, visible_triangle_ids)
            np.testing.assert_equal(counts, visible_triangle_counts)
            res = True
        except AssertionError as err:
            res = False
            print(err)
        self.assertTrue(res)

    def test_rasterize(self):
        triangle = np.array([
            [1, 1, 1],
            [0.99, 1, 1],
            [1, 1, 0.99]
        ])

        expected = np.array([[610, 190],
                             [610, 191],
                             [610, 192],
                             [610, 193],
                             [610, 194],
                             [611, 190],
                             [611, 191],
                             [611, 192],
                             [611, 193],
                             [612, 190],
                             [612, 191],
                             [612, 192],
                             [613, 190],
                             [613, 191],
                             [614, 190]])

        try:
            np.testing.assert_equal(self.renderer.rasterize(triangle, self.cam), expected)
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
        p_inside2 = [2, 0]
        p_outside = [2, 2]

        self.assertTrue(self.renderer.barycentric(p_inside1, triangle))
        self.assertTrue(self.renderer.barycentric(p_inside2, triangle))
        self.assertFalse(self.renderer.barycentric(p_outside, triangle))

        triangle2 = np.array([
            [12, 16],
            [27, 16],
            [27, 3]
        ])

        p_inside3 = [27, 3]

        self.assertTrue(self.renderer.barycentric(p_inside3, triangle2))

        triangle3 = np.array([
            [13, 16],
            [27, 4],
            [13, 4]
        ])

        p_inside4 = [18, 4]
        p_inside5 = [13, 16]

        self.assertTrue(self.renderer.barycentric(p_inside4, triangle3))
        self.assertTrue(self.renderer.barycentric(p_inside5, triangle3))

    def test_get_bounding_box_coords_for_triangle(self):
        triangle = np.array([
            [1, 0],
            [2, 0],
            [1, 2]
        ])

        expected = np.array([
            [1, 0],
            [1, 1],
            [1, 2],
            [2, 0],
            [2, 1],
            [2, 2]
        ])

        try:
            np.testing.assert_equal(self.renderer.get_bounding_box_coords_for_triangle(triangle), expected)
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
            [0, 0],
            [0, 1],
            [0, 2],
            [1, 0],
            [1, 1],
            [1, 2],
            [2, 0],
            [2, 1],
            [2, 2]
        ])

        try:
            np.testing.assert_equal(self.renderer.get_bounding_box_coords_for_triangle(triangle_2), expected_2)
            res = True
        except AssertionError as err:
            res = False
            print(err)
        self.assertTrue(res)