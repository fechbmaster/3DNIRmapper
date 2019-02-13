import string

import numpy as np

from nirmapper.exceptions import TextureError
from nirmapper.renderer.camera import Camera


class Texture(object):
    __visible_vertices = []
    __vert_indices = []
    __uv_coords = []
    __uv_indices = []
    __normal_indices = []
    counts = []
    vis_triangle_ids = []

    def __init__(self, texture_path: string, cam: Camera):
        self.texture_path = texture_path
        self.cam = cam

    @property
    def visible_vertices(self) -> np.ndarray:
        return self.__visible_vertices

    @visible_vertices.setter
    def visible_vertices(self, visible_vertices: np.ndarray):
        if visible_vertices is None or visible_vertices.size == 0:
            self.__visible_vertices = []
            return
        visible_vertices = self.__reshape(visible_vertices, 3)
        self.__visible_vertices = visible_vertices

    @property
    def verts_indices(self) -> np.ndarray:
        return self.__vert_indices

    @verts_indices.setter
    def verts_indices(self, verts_indices: np.ndarray):
        if verts_indices is None or verts_indices.size == 0:
            self.__vert_indices = []
            return
        visible_vertices = self.__reshape(verts_indices, 3)
        self.__vert_indices = visible_vertices

    @property
    def uv_coords(self) -> np.ndarray:
        return self.__uv_coords

    @uv_coords.setter
    def uv_coords(self, uv_coords: np.ndarray):
        if uv_coords is None or uv_coords.size == 0:
            self.__uv_coords = []
            return
        # Reshape to get coords
        uv_coords = self.__reshape(uv_coords, 2)
        self.__uv_coords = uv_coords

    @property
    def uv_indices(self) -> np.ndarray:
        return self.__uv_indices

    @uv_indices.setter
    def uv_indices(self, uv_indices: np.ndarray):
        if uv_indices is None or uv_indices.size == 0:
            self.__uv_indices = []
            return
        self.__uv_indices = uv_indices

    @property
    def normal_indices(self) -> np.ndarray:
        return self.__normal_indices

    @normal_indices.setter
    def normal_indices(self, normal_indices: np.ndarray):
        if normal_indices is None or normal_indices.size == 0:
            self.__normal_indices = []
            return
        # Reshape to get coords
        normal_indices = self.__reshape(normal_indices, 3)
        self.__normal_indices = normal_indices

    @staticmethod
    def __reshape(array: np.ndarray, vert_length: int) -> np.ndarray:
        """
        Method reshapes an array depending on its length by giving the vertices length.

        :param array: The array that should be reshaped.
        :param vert_length: The length of the vectors.
        :return: The reshaped array.
        """
        try:
            return array.reshape([(array.size // vert_length), vert_length])
        except ValueError as e:
            raise TextureError(e)
