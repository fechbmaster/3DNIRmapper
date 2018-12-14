from typing import List

import numpy as np
import pywavefront
from collada import *
from collada import source

from nirmapper.exceptions import WavefrontError, ModelError


class Model(object):
    """This is the model defining class.
    """

    __normals = np.array([])
    __vertices = np.array([])
    __uv_coords = np.array([])

    def __init__(self, vertices: np.ndarray, indices: np.ndarray = None, normals: np.ndarray = None,
                 uv_coords: np.ndarray = None):
        self.vertices = vertices
        self.indices = np.array([]) if indices is None else indices
        self.normals = normals
        self.uv_coords = uv_coords

    @property
    def vertices(self) -> np.ndarray:
        return self.__vertices

    @vertices.setter
    def vertices(self, vertices: np.ndarray):
        # Reshape to get vertices
        vertices.reshape([(vertices.size // 3), 3])
        if self.normals.size != 0:
            if vertices.shape != self.normals.shape:
                raise ModelError("Invalid vertices shape. Shape is not matching the defined normals shape.")
        if self.uv_coords.size != 0:
            if vertices.shape[0] != self.uv_coords.shape[0]:
                raise ModelError("Invalid vertices shape. Length is not matching the defined uv coordinates length.")
        self.__vertices = vertices

    @property
    def normals(self) -> np.ndarray:
        return self.__normals

    @normals.setter
    def normals(self, normals: np.ndarray):
        if normals is None:
            return
        # Reshape to get vertices
        normals.reshape([(normals.size // 3), 3])
        if self.vertices.size != 0:
            if normals.shape != self.vertices.shape:
                raise ModelError("Invalid normals shape. Shape is not matching the defined vertices shape.")
        if self.uv_coords.size != 0:
            if normals.shape[0] != self.uv_coords.shape[0]:
                raise ModelError("Invalid normals shape. Length is not matching the defined uv coordinates length.")
        self.__normals = normals

    @property
    def uv_coords(self) -> np.ndarray:
        return self.__uv_coords

    @uv_coords.setter
    def uv_coords(self, uv_coords: np.ndarray):
        if uv_coords is None:
            return
        # Reshape to get coords
        uv_coords.reshape([(uv_coords.size // 2), 2])
        if self.vertices.size != 0:
            if uv_coords.shape[0] != self.vertices.shape[0]:
                raise ModelError("Invalid uv coordinates shape. Length is not matching the defined normals length.")
        if self.uv_coords.size != 0:
            if uv_coords.shape[0] != self.normals.shape[0]:
                raise ModelError("Invalid uv coordinates shape. Length is not matching the defined normals length.")
        self.__uv_coords = uv_coords

    def generate_indices(self) -> np.ndarray:
        ind_len = 0
        dim_ind = 0
        if len(self.vertices) != 0:
            dim_ind += 1
            ind_len = np.shape(self.vertices)[0]
        if len(self.normals) != 0:
            dim_ind += 1
            if ind_len == 0: ind_len = np.shape(self.normals)[0]
        if len(self.uv_coords) != 0:
            dim_ind += 1
            if ind_len == 0: ind_len = np.shape(self.uv_coords)[0]
        if dim_ind > 0 and ind_len > 0:
            return np.indices((ind_len, dim_ind))[0]


class ColladaCreator(object):
    """A creator class for Collada files.
    """

    @staticmethod
    def create_collada_from_model(model: Model, texture_path: str, output_path: str) -> None:
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
        # todo: check if this can be converted to tuple
        mat_map = material.Map(sampler2d, "UVSET0")

        effect = material.Effect("effect0", [surface, sampler2d], "lambert", emission=(0.0, 0.0, 0.0, 1),
                                 ambient=(0.0, 0.0, 0.0, 1), diffuse=mat_map, transparent=mat_map, transparency=0.0,
                                 double_sided=True)
        mat = material.Material("material0", "mymaterial", effect)

        mesh.effects.append(effect)
        mesh.materials.append(mat)
        mesh.images.append(image)

        vert_src = source.FloatSource("cubeverts-array", np.array(model.vertices), ('X', 'Y', 'Z'))
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
        node = scene.Node("Cube", children=[geomnode])

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
            formatter = VertexFormatter(format_string)

            # V3F is mandatory
            if 'V3F' not in formatter.formats:
                raise WavefrontError("Position Vertices not found in .obj file")

            # Contains all vertices no matter if T2F, C3F, N3F or V3F
            all_verts = np.array(obj_material.vertices)

            vertices = formatter.get_verts_by_format(all_verts, 'V3F')
            model = Model(vertices)

            # Normals are optional
            if 'N3F' in formatter.formats:
                normals = formatter.get_verts_by_format(all_verts, 'N3F')
                model.normals = np.array(normals)

            indices = model.generate_indices()
            model.indices = indices
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


class VertexFormatter(object):
    # These are the valid formats with their vertices length
    __valid_vertex_formats = {
        'T2F': 2,
        'C3F': 3,
        'N3F': 3,
        'V3F': 3
    }

    def __init__(self, format_string: str):
        self.format_string: str = format_string
        self.formats: List[str] = format_string.split("_")
        self.__validate_formats(self.formats)

    def get_verts_by_format(self, verts: np.ndarray, format: str):
        if format not in self.formats:
            return np.array([])

        seq = self.get_vert_lengths()
        start_index = self.get_start_index_for_format(format)
        length = self.get_length_for_format(format)

        return np.array([verts[i:i + length] for i in range(start_index, len(verts), seq.sum())])

    def get_vert_lengths(self):
        format_values = np.zeros(len(self.formats), dtype=np.int)
        # Convert to enum values and build up format_values array
        for idx, vert_format in enumerate(self.formats):
            format_values[idx] = self.__valid_vertex_formats.get(self.formats[idx])

        return format_values

    def get_start_index_for_format(self, format):
        """
        Builds an index sequence for values by adding the previous element for every element.

        Example: The sequence [2, 3, 3] results to [2, 5, 8]
        :return:
        """
        seq = self.get_vert_lengths()  # [2, 3, 3]
        index = self.get_index_for_format(format)

        start_index = seq[:index].sum()

        return start_index

    def get_length_for_format(self, format):
        return self.__valid_vertex_formats.get(format)

    def get_index_for_format(self, format):
        return self.formats.index(format)

    def __validate_formats(self, formats):
        for format in formats:
            if format not in self.__valid_vertex_formats:
                raise WavefrontError("The specified vertex format sequence is invalid!")
