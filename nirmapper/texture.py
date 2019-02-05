import string

import numpy as np

from nirmapper import Camera, Model


class Texture(object):
    z_buffer: np.ndarray

    def __init__(self, text_id: int, texture_path: string, cam: Camera):
        self.id = text_id
        self.texture_path = texture_path
        self.cam = cam

    def check_occlusion_for_model(self, model: Model):
        self.create_z_buffer(model)
        return []

    def create_z_buffer(self, model: Model):
        width = self.cam.resolution_x
        height = self.cam.resolution_y

        z_buffer = np.zeros([width, height])
        triangles = model.get_triangles()

        # for triangle in triangles:

    def get_pixels_for_triangle(self, vertices: np.ndarray) -> np.ndarray:
        if vertices.shape != (3, 3):
            raise ValueError("Given triangle must be of shape (3, 3).")

        # Get texture coords for vertice
        text_coords = self.cam.get_pixel_coords_for_vertices(vertices)
        bounding_box = self.get_bounding_box_coords_for_triangle(text_coords)

        included_pixels = []
        for pixel in bounding_box.reshape(bounding_box.size // 2, 2):
            if self.pixel_is_included_in_triangle(text_coords, pixel):
                included_pixels.append(pixel)

        return np.array(included_pixels)

    def pixel_is_included_in_triangle(self, text_coords: np.ndarray, pixel) -> bool:
        if text_coords.shape != (3, 2):
            raise ValueError("Triangle is in wrong shape. Shape must be (3, 2).")
        if len(pixel) != 2:
            raise ValueError("Pixel must have two coordinates.")

        # first sort triangle coords by their x value
        coords = self.__sort_triangles_by_x_coord(text_coords)

        inside = True

        inside &= self.__edge_function(coords[0], coords[1], pixel)
        inside &= self.__edge_function(coords[1], coords[2], pixel)
        inside &= self.__edge_function(coords[2], coords[0], pixel)

        return inside

    @staticmethod
    def get_bounding_box_coords_for_triangle(text_coords: np.ndarray) -> np.ndarray:
        min_x = np.amin(text_coords[:, 0])
        max_x = np.amax(text_coords[:, 0])
        min_y = np.amin(text_coords[:, 1])
        max_y = np.amax(text_coords[:, 1])

        x = np.arange(min_x, max_x + 1)
        y = np.arange(min_y, max_y + 1)

        box = np.zeros((y.size, x.size, 2))

        # todo: maybe don't convert to a 3d array - actually not needed
        for row_idx, row in enumerate(box):
            for col_idx, column in enumerate(row):
                box[row_idx, col_idx] = [x[col_idx], y[row_idx]]

        return box

    @staticmethod
    def __sort_triangles_by_x_coord(triangles) -> np.ndarray:
        return triangles[triangles[:, 0].argsort()]

    @staticmethod
    def __edge_function(v1, v2, p) -> bool:
        """
        The edge function determines if a point p is right, left or on line of a edge
        defined by two texture coordinates v1 and v2.

        E(P) > 0 if P is to the "right" side
        E(P) = 0 if P is exactly on the line
        E(P) < 0 if P is to the "left " side

        :param v1: first texture coordinate
        :param v2: second texture coordinate
        :param p: point to check
        :return bool: returns true if point is on or right of the edge
        """
        return (p[0] - v1[0]) * (v2[1] - v1[1]) - (p[1] - v1[1]) * (v2[0] - v1[0]) >= 0
