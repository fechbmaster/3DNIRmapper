from enum import Enum
from collada import *
from collada import source
import numpy as np
import pywavefront

from nirmapper.exceptions import WavefrontError


class Model(object):
    """This is the model defining class.
    """

    def __init__(self, vertices, indices, normals=None, uv_coords=None):
        self.vertices = vertices
        self.indices = indices
        self.normals = [] if normals is None else normals
        self.uv_coords = [] if uv_coords is None else uv_coords


class ColladaCreator(object):
    """A creator class for Collada files.
    """

    @staticmethod
    def create_collada_from_model(model: Model, texture_path, output_path):
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
        map = material.Map(sampler2d, "UVSET0")

        effect = material.Effect("effect0", [surface, sampler2d], "lambert", emission=(0.0, 0.0, 0.0, 1),
                                 ambient=(0.0, 0.0, 0.0, 1), diffuse=map, transparent=map, transparency=0.0,
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


class WavefrontImporter(object):
    """Class for importing Wavefront .obj files.

    """

    # todo: Make this a better data structure
    vertex_formats = {
        'T2F': 2,
        'C3F': 3,
        'N3F': 3,
        'V3F': 3
    }

    @staticmethod
    def import_obj_as_model_list(file_path):
        """
        Method converts a .obj file to a model list for post processing.

        :param file_path: The absolute path to the .obj file.
        :return: List of models of type Model.
        """
        scene = pywavefront.Wavefront(file_path, cache=True)
        models = []

        for name, material in scene.materials.items():
            # Contains the vertex format (string) such as "T2F_N3F_V3F"
            # T2F, C3F, N3F and V3F may appear in this string
            # Only V3F and N3F is needed for model creation
            format_string = material.vertex_format
            formats = format_string.split("_")
            format_values = np.zeros(len(formats), dtype=np.int)

            # Convert to enum values and build up format_values array
            for idx, format in enumerate(formats):
                format_values[idx] = WavefrontImporter.vertex_formats.get(formats[idx])

            # V3F is mandatory
            if 'V3F' in formats:
                vert_index = formats.index('V3F')
            else:
                raise WavefrontError("Position Vertices not found in .obj file")

            # Contains all vertices no matter if T2F, C3F, N3F or V3F
            all_verts = material.vertices

            # Build the sequence described by the vertex format
            seq = WavefrontImporter.__build_seq(format_values)
            split_seq = WavefrontImporter.__built_split_seq(seq, len(all_verts))
            sorted_verts = np.split(all_verts, split_seq)[:-1]

            vertices = sorted_verts[vert_index::len(formats)]
            vertices = np.concatenate(vertices, axis=None)
            indices = np.arange(len(vertices))
            model = Model(vertices, indices)

            # Normals are optional
            if 'N3F' in formats:
                norm_index = formats.index('N3F')
                normals = all_verts[norm_index::len(formats)]
                normals = np.concatenate(normals, axis=None)
                indices = np.array([indices, indices]).T
                model.indices = indices
                model.normals = normals

            models.append(model)

        return models

    @staticmethod
    def __built_split_seq(seq, length):
        """
        Method builds a split sequence out of a small sequence by a given length.

        Example: The Sequence [2,5,8] results into a extended sequence of [ 2  5  8 10 13 16 18 21 24]
        by the given length of 24.


        :param seq: The sequence that should be extended.
        :param length: The length the sequence should be extended to.
        :return: The extended sequence.
        """
        if (length % seq[-1]) != 0:
            raise WavefrontError("Can't divide vertices list to the sub-elements defined by the format string.")

        split_seq = seq

        while (split_seq[-1] + seq[-1]) <= length:
            d = split_seq + seq[-1]
            split_seq = np.append(seq, d)

        return split_seq

    @staticmethod
    def __build_seq(format_values):
        """
        Builds an index sequence for values by adding the previous element for every element.

        Example: The sequence [2, 3, 3] results to [2, 5, 8]

        :param format_values:
        :return:
        """
        seq = np.copy(format_values)
        for x in range(1, len(format_values)):
            seq[x] = seq[x] + seq[x - 1]

        return seq
