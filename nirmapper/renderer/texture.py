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
    counts = np.array([], dtype=int)
    vis_triangle_ids = []
    duplicate_triangle_ids = np.array([], dtype=int)

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

    def remove_duplicate_data_at_triangle_id(self, triangle_id: int):
        if triangle_id in self.duplicate_triangle_ids:
            # Delete elements
            idx = np.where(self.vis_triangle_ids == triangle_id)[0]
            self.__delete_triangle_at_index(idx)

    def __delete_triangle_at_index(self, tri_idx: int):
        # Delete verts
        indices = np.arange(tri_idx * 3, (tri_idx + 1) * 3)
        self.visible_vertices = np.delete(self.visible_vertices, indices, axis=0)
        # Delete vert indices
        self.verts_indices = np.delete(self.verts_indices, tri_idx, axis=0)
        # Delete uvs
        self.uv_coords = np.delete(self.uv_coords, indices, axis=0)
        # Re-arange uv indices
        self.uv_indices = np.arange(np.size(self.uv_coords) // 2)
        # Delete normal indices
        self.normal_indices = np.delete(self.normal_indices, tri_idx, axis=0)
        # Delete triangle id
        self.vis_triangle_ids = np.delete(self.vis_triangle_ids, tri_idx)


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
