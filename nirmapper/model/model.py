import numpy as np

from nirmapper.exceptions import ModelError


class Model(object):
    """This is the model defining class.
    """

    __normals = []  # shape(x, 3) => Normals
    __vertices = []  # shape(x, 3) => All Vertices
    __uv_coords = []  # shape(x, 2) => Texture coordinates
    __indices = []  # shape(x, 3) => Defines the triangles by indexing 3 vertices by triangle
    __uv_indices = []  # shape(x, 3) => Indexing of the texture coordinates
    __normal_indices = []  # shape(x, 3) => Indexing of the normals

    def __init__(self, vertices: np.ndarray = None, normals: np.ndarray = None, uv_coords: np.ndarray = None):
        self.vertices = vertices
        self.normals = normals
        self.uv_coords = uv_coords

    @property
    def vertices(self) -> np.ndarray:
        return self.__vertices

    @vertices.setter
    def vertices(self, vertices: np.ndarray):
        if vertices is None or vertices.size == 0:
            self.__vertices = []
            return
        vertices = self.__reshape(vertices, 3)
        self.__vertices = vertices

    @property
    def normals(self) -> np.ndarray:
        return self.__normals

    @normals.setter
    def normals(self, normals: np.ndarray):
        if normals is None or normals.size == 0:
            self.__normals = []
            return
        # Reshape to get vertices
        normals = self.__reshape(normals, 3)
        self.__normals = normals

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
    def indices(self) -> np.ndarray:
        return self.__indices

    @indices.setter
    def indices(self, indices: np.ndarray):
        if indices is None or indices.size == 0:
            self.__indices = []
            return
        # Reshape to get coords
        indices = self.__reshape(indices, 3)
        self.__indices = indices

    @property
    def uv_indices(self) -> np.ndarray:
        return self.__uv_indices

    @uv_indices.setter
    def uv_indices(self, uv_indices: np.ndarray):
        if uv_indices is None or uv_indices.size == 0:
            self.__uv_indices = []
            return
        # Reshape to get coords
        uv_indices = self.__reshape(uv_indices, 3)
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
            raise ModelError(e)
