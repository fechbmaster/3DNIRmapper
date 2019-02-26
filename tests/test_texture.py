from unittest import TestCase

import numpy as np

from nirmapper.camera import Camera
from nirmapper.model import Texture


class TestTexture(TestCase):

    def setUp(self):
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

        texture = Texture('/fake_path', cam1)
        texture.visible_vertices = np.array([
            [1, 1, -1],
            [-1, 1, 1],
            [1, 1, 1],
            [1, 1, -1],
            [-1, 1, -1],
            [-1, 1, 1]
        ])

        texture.normal_indices = np.array(
            [[0, 7, 4],
             [0, 3, 7]
             ])

        texture.uv_coords = np.array(
            [[0.31770833, 0.17592593],
             [0.68229167, 0.82407407],
             [0.31770833, 0.82407407],
             [0.31770833, 0.17592593],
             [0.68229167, 0.17592593],
             [0.68229167, 0.82407407]]
        )
        texture.arange_uv_indices()

        texture.verts_indices = [0, 7, 4, 0, 3, 7]
        texture.counts = [667, 665]
        texture.vis_triangle_indices = [5, 11]
        texture.duplicate_triangle_indices = [11]

        self.texture = texture

    def test_remove_triangle_with_index(self):
        expected_visible_vertices = np.array([
            [1, 1, -1],
            [-1, 1, 1],
            [1, 1, 1]
        ])
        expected_uv_coords = np.array(
            [[0.31770833, 0.17592593],
             [0.68229167, 0.82407407],
             [0.31770833, 0.82407407]]
        )

        expected_vis_vert_indices = np.array([[0, 7, 4]])
        expected_normal_indices = np.array([[0, 7, 4]])
        expected_uv_indices = np.array([0, 1, 2])

        expected_counts = [667]
        expected_vis_triangle_ids = [5]
        expected_dup_triangle_ids = []

        self.texture.remove_triangle_with_index(11)

        try:
            np.testing.assert_equal(self.texture.visible_vertices, expected_visible_vertices)
            np.testing.assert_equal(self.texture.verts_indices, expected_vis_vert_indices)
            np.testing.assert_equal(self.texture.normal_indices, expected_normal_indices)
            np.testing.assert_equal(self.texture.uv_coords, expected_uv_coords)
            np.testing.assert_equal(self.texture.uv_indices, expected_uv_indices)
            np.testing.assert_equal(self.texture.counts, expected_counts)
            np.testing.assert_equal(self.texture.vis_triangle_indices, expected_vis_triangle_ids)
            np.testing.assert_equal(self.texture.duplicate_triangle_indices, expected_dup_triangle_ids)
            res = True
        except AssertionError as err:
            res = False
            print(err)
        self.assertTrue(res)
