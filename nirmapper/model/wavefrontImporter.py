from enum import Enum
from typing import List

import numpy as np
import pywavefront

from nirmapper.model.model import Model
from nirmapper.exceptions import WavefrontError
from nirmapper.utils import quaternion_matrix


class IndicesFormat(Enum):
    """Enum that holds the valid formats for the indices.

    """
    T2F = 1,
    #C3F = 2,   # not needed here
    N3F = 3,
    V3F = 4

    @staticmethod
    def get_indices_formats_from_string(format_str: str):
        """
        Converts a string 'T2F_N3F' to a sequence of enum values.

        :param format_str: String of the format.
        :return: List of enum formats.
        """
        formats = []
        format_str_array = format_str.split("_")
        for format_str in format_str_array:
            formats.append(IndicesFormat[format_str])

        return formats

    @staticmethod
    def get_length_for_format(ind_format):
        """
        Get the length for an format.

        :param ind_format: The format.
        :return: The length of the format.
        """
        if ind_format == IndicesFormat.T2F:
            return 2
        else:
            return 3


class Wavefront(object):
    """Class for importing Wavefront .obj files.

    The class is constructed in a way, that it can be used by calling the static method to immediately getting the Model
    objects, or by creating a Wavefront object, that holds more information.

    """

    def __init__(self, file_path: str, cache: bool = True):
        self.scene = pywavefront.Wavefront(file_path, cache=cache, create_materials=True)
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
            formatter = IndicesFormatter(formats)

            # V3F is mandatory
            if IndicesFormat.V3F not in formatter.formats:
                raise WavefrontError("Position Vertices not found in .obj file")

            # Contains all vertices no matter if T2F, C3F, N3F or V3F
            all_verts = np.array(obj_material.vertices)
            rotation = quaternion_matrix([0.707, 0.707, 0, 0.0])

            # Create empty model
            model = Model()

            for ind_format in formats:
                coords = formatter.get_coords_by_format(all_verts, ind_format)
                # rotate vertices by x=90° degrees to get into internal coord system
                # todo: make this better!!!!!!
                if ind_format != IndicesFormat.T2F:
                    coords = coords.reshape(coords.size // 3, 3)
                    coords = np.array([rotation.dot(x) for x in coords])
                    # There are no triangle describing indices so we have to generate the indices
                    indices = formatter.generate_indices(coords.size // 3)
                else:
                    # There are no triangle describing indices so we have to generate the indices
                    indices = formatter.generate_indices(coords.size // 2)
                if ind_format == IndicesFormat.V3F:
                    model.vertices = coords
                    model.indices = indices
                elif ind_format == IndicesFormat.N3F:
                    model.normals = coords
                    model.normal_indices = indices
                elif ind_format == IndicesFormat.T2F:
                    model.uv_coords = coords
                    model.uv_indices = indices

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
        cust_scene = pywavefront.Wavefront(file_path, cache=cache, create_materials=True)
        model = Wavefront.__import_obj_as_model_list_from_scene(cust_scene)

        return model


class IndicesFormatter(object):
    """ A formatter that gives functionality to split a list or array of vertices depending on a format sequence.

    """

    def __init__(self, formats: List[IndicesFormat]):
        self.formats = formats

    def get_coords_by_format(self, verts, ind_format: IndicesFormat):
        """
        Method splits up an array of vertices into an array of specific vertices described by an format and a
        sequence.

        :param verts: The vertices to split.
        :param ind_format: The format the vertices should be split.
        :return: The split vertices depending on the sequence.
        """
        seq = self.__get_coord_lengths()
        start_index = self.__get_start_index_for_format(ind_format)
        length = IndicesFormat.get_length_for_format(ind_format)

        return np.array([verts[i:i + length] for i in range(start_index, len(verts), seq.sum())])

    def __get_coord_lengths(self):
        """
        Method returns the length for the format sequence.

        :return: The length sequence.
        """
        format_values = np.zeros(len(self.formats), dtype=np.int)
        # Convert to enum values and build up format_values array
        for idx, vert_format in enumerate(self.formats):
            format_values[idx] = IndicesFormat.get_length_for_format(vert_format)

        return format_values

    def __get_start_index_for_format(self, ind_format):
        """
        Builds an index sequence for values by adding the previous element for every element.

        Example: The sequence [2, 3, 3] results to [2, 5, 8]
        :return:
        """
        seq = self.__get_coord_lengths()  # [2, 3, 3]
        index = self.__get_index_for_format(ind_format)

        start_index = seq[:index].sum()

        return start_index

    def __get_index_for_format(self, ind_format):
        return self.formats.index(ind_format)

    # def get_indices_for_format(self, ind_format: IndicesFormat, indices: np.ndarray) -> np.ndarray:
    #     """
    #     Gets the indices that describes coordinates by a format.
    #
    #     :param indices: The indices list
    #     :param ind_format: The format of the indices.
    #     :return: The indices for the format.
    #     """
    #     index = self.formats.index(ind_format)
    #
    #     return indices[:, [index]]

    @staticmethod
    def generate_indices(length: int) -> np.ndarray:
        """
        Generates indices depending on the given vertices.

        :return: An array of indices, The format of the indices.
        """
        return np.arange(0, length)
