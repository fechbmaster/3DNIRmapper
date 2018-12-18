from enum import Enum
from typing import List, Union, Tuple

import numpy as np
import pywavefront
from collada import *
from collada import source

from nirmapper.exceptions import WavefrontError, ModelError


class IndicesFormat(Enum):
    T2F = 1,
    # C3F - actually not needed but supported
    C3F = 2,
    N3F = 3,
    V3F = 4

    @staticmethod
    def get_indices_formats_from_string(format_str: str):
        formats: List[IndicesFormat] = []
        format_str_array = format_str.split("_")
        for format_str in format_str_array:
            formats.append(IndicesFormat[format_str])

        return formats

    @staticmethod
    def get_length_for_format(ind_format):
        if ind_format == IndicesFormat.T2F:
            return 2
        else:
            return 3


class Model(object):
    """This is the model defining class.
    """

    __normals = np.array([])
    __vertices = np.array([])
    __uv_coords = np.array([])
    __indices = np.array([])

    def __init__(self, obj_vertices: np.ndarray = None, normals: np.ndarray = None,
                 uv_coords: np.ndarray = None):
        self.obj_vertices = obj_vertices
        self.normals = normals
        self.uv_coords = uv_coords
        self.indices_format: List[IndicesFormat] = []

    @property
    def obj_vertices(self) -> np.ndarray:
        return self.__vertices

    @obj_vertices.setter
    def obj_vertices(self, vertices: np.ndarray):
        if vertices is None or vertices.size == 0:
            self.__vertices = np.array([])
            return
        vertices = self.__reshape(vertices, 3)
        self.__vertices = vertices

    @property
    def normals(self) -> np.ndarray:
        return self.__normals

    @normals.setter
    def normals(self, normals: np.ndarray):
        if normals is None or normals.size == 0:
            self.__normals = np.array([])
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
            self.__uv_coords = np.array([])
            return
        # Reshape to get coords
        uv_coords = self.__reshape(uv_coords, 2)
        self.__uv_coords = uv_coords

    @property
    def indices(self) -> np.ndarray:
        return self.__indices

    def set_indices(self, indices: np.ndarray, ind_format: Union[str, List[IndicesFormat]]):
        if indices is None or indices.size == 0:
            self.__indices = np.array([])
            return
        if type(ind_format) is str:
            ind_format = IndicesFormat.get_indices_formats_from_string(ind_format)

        dim_ind = 1
        if len(self.normals) != 0:
            dim_ind += 1
        if len(self.uv_coords) != 0:
            dim_ind += 1
        # Reshape to get coords
        indices = self.__reshape(indices, dim_ind)
        self.__indices = indices
        self.indices_format = ind_format

    def generate_indices(self) -> Tuple[np.ndarray, List[IndicesFormat]]:
        ind_len = 0
        dim_ind = 0
        ind_format = []
        if self.obj_vertices.size != 0:
            ind_len = np.shape(self.obj_vertices)[0]
            dim_ind += 1
            ind_format.append(IndicesFormat.V3F)
        if self.normals.size != 0:
            if ind_len == 0:
                ind_len = np.shape(self.normals)[0]
            dim_ind += 1
            ind_format.append(IndicesFormat.N3F)
        if self.uv_coords.size != 0:
            if ind_len == 0:
                ind_len = np.shape(self.uv_coords)[0]
            dim_ind += 1
            ind_format.append(IndicesFormat.T2F)
        if ind_len == 0 and dim_ind == 0:
            return np.array([]), ind_format
        return np.indices((ind_len, dim_ind))[0], ind_format

    @staticmethod
    def __reshape(array: np.ndarray, vert_length: int):
        try:
            return array.reshape([(array.size // vert_length), vert_length])
        except ValueError as e:
            raise ModelError(e)

    def get_indices_for_format(self, ind_format: IndicesFormat):
        if ind_format not in self.indices_format:
            print("Searched for %s format indices, but it is not defined" % ind_format)
            return np.array([])
        index = self.indices_format.index(ind_format)

        return self.indices[:, [index]]

    def set_vertices_by_ind_format(self, vertices, ind_format: IndicesFormat):
        if ind_format == IndicesFormat.V3F:
            self.obj_vertices = vertices
        elif ind_format == IndicesFormat.N3F:
            self.normals = vertices
        elif ind_format == IndicesFormat.T2F:
            self.uv_coords = vertices


class ColladaCreator(object):
    """A creator class for Collada files.
    """

    @staticmethod
    def create_collada_from_model(model: Model, texture_path: str, output_path: str, node_name: str) -> None:
        """
        Create a Collada file out of an modell and a texture.

        :param model: The model
        :param texture_path: Path of the texture
        :param output_path: Path where collada file should be stored
        """
        mesh = Collada()

        # needed for texture
        image = material.CImage("material_0-image", texture_path)
        surface = material.Surface("material_0-image-surface", image)
        sampler2d = material.Sampler2D("material_0-image-sampler", surface)
        mat_map: material.Map = material.Map(sampler2d, "UVSET0")

        effect = material.Effect("effect0", [surface, sampler2d], "lambert", emission=(0.0, 0.0, 0.0, 1),
                                 ambient=(0.0, 0.0, 0.0, 1), diffuse=mat_map, transparent=mat_map, transparency=0.0,
                                 double_sided=True)
        mat = material.Material("material0", "mymaterial", effect)

        mesh.effects.append(effect)
        mesh.materials.append(mat)
        mesh.images.append(image)

        vert_src = source.FloatSource("cubeverts-array", np.array(model.obj_vertices), ('X', 'Y', 'Z'))
        normal_src = source.FloatSource("cubenormals-array", np.array(model.normals), ('X', 'Y', 'Z'))
        uv_src = source.FloatSource("cubeuv_array", np.array(model.uv_coords), ('S', 'T'))

        geom = geometry.Geometry(mesh, "geometry0", "mycube", [vert_src, normal_src, uv_src])
        input_list = source.InputList()
        input_list.addInput(0, 'VERTEX', "#cubeverts-array")
        input_list.addInput(1, 'TEXCOORD', "#cubeuv_array", set="0")
        input_list.addInput(2, 'NORMAL', "#cubenormals-array")

        triset = geom.createTriangleSet(model.indices, input_list, "materialref")
        geom.primitives.append(triset)
        mesh.geometries.append(geom)

        matnode = scene.MaterialNode("materialref", mat, inputs=[])
        geomnode = scene.GeometryNode(geom, [matnode])
        node = scene.Node(node_name, children=[geomnode])

        myscene = scene.Scene("myscene", [node])
        mesh.scenes.append(myscene)
        mesh.scene = myscene

        mesh.write(output_path)


class Wavefront(object):
    """Class for importing Wavefront .obj files.

    The class is constructed in a way, that it can be used by calling the static method to immediately getting the Model
    objects, or by creating a Wavefront object, that holds more information.

    """

    def __init__(self, file_path: str, cache: bool = True):
        self.scene = pywavefront.Wavefront(file_path, cache=cache)
        self.models = Wavefront.__import_obj_as_model_list_from_scene(self.scene)

    @staticmethod
    def __import_obj_as_model_list_from_scene(cust_scene: pywavefront.Wavefront) -> List[Model]:
        """
        Method converts a .obj file to a model list for post processing.

        :param cust_scene: A predefined scene.
        :return: List of models of type Model.
        """
        models = []
        for name, obj_material in cust_scene.materials.items():
            # Contains the vertex format (string) such as "T2F_N3F_V3F"
            # T2F, C3F, N3F and V3F may appear in this string
            # Only V3F and N3F is needed for model creation
            format_string = obj_material.vertex_format
            formats = IndicesFormat.get_indices_formats_from_string(format_string)
            formatter = VertexIndicesFormatter(formats)

            # V3F is mandatory
            if IndicesFormat.V3F not in formatter.formats:
                raise WavefrontError("Position Vertices not found in .obj file")

            # Contains all vertices no matter if T2F, C3F, N3F or V3F
            all_verts = np.array(obj_material.vertices)

            # Create empty model
            model = Model()

            for ind_format in formats:
                vertices = formatter.get_verts_by_format(all_verts, ind_format)
                model.set_vertices_by_ind_format(vertices, ind_format)

            indices, ind_format = model.generate_indices()
            model.set_indices(indices, ind_format)
            models.append(model)

        return models

    @staticmethod
    def import_obj_as_model_list(file_path: str, cache: bool = True) -> List[Model]:
        """
        Method converts a .obj file to a model list for post processing.

        :param file_path: The absolute path to the .obj file.
        :param cache: Set caching off or on.
        :return: List of models of type Model.
        """
        cust_scene = pywavefront.Wavefront(file_path, cache=cache)
        model = Wavefront.__import_obj_as_model_list_from_scene(cust_scene)

        return model


class VertexIndicesFormatter(object):
    # These are the valid formats with their vertices length

    def __init__(self, formats: List[IndicesFormat]):
        self.formats = formats

    def get_verts_by_format(self, verts: np.ndarray, ind_format: IndicesFormat):
        # if format in self.formats:
        #     return np.array([])

        seq = self.get_vert_lengths()
        start_index = self.get_start_index_for_format(ind_format)
        length = IndicesFormat.get_length_for_format(ind_format)

        return np.array([verts[i:i + length] for i in range(start_index, len(verts), seq.sum())])

    def get_vert_lengths(self):
        format_values = np.zeros(len(self.formats), dtype=np.int)
        # Convert to enum values and build up format_values array
        for idx, vert_format in enumerate(self.formats):
            format_values[idx] = IndicesFormat.get_length_for_format(vert_format)

        return format_values

    def get_start_index_for_format(self, ind_format):
        """
        Builds an index sequence for values by adding the previous element for every element.

        Example: The sequence [2, 3, 3] results to [2, 5, 8]
        :return:
        """
        seq = self.get_vert_lengths()  # [2, 3, 3]
        index = self.get_index_for_format(ind_format)

        start_index = seq[:index].sum()

        return start_index

    def get_index_for_format(self, ind_format):
        return self.formats.index(ind_format)
