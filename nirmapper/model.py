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

    @staticmethod
    def import_obj_as_model_list(file_path):
        """
        Method converts a .obj file to a model list for post processing.

        :param file_path: The absolute path to the .obj file.
        :return: List of models of type Model
        """
        scene = pywavefront.Wavefront(file_path, cache=True)
        models = []

        for name, material in scene.materials.items():
            # Contains the vertex format (string) such as "T2F_N3F_V3F"
            # T2F, C3F, N3F and V3F may appear in this string
            # Only V3F and N3F is needed for model creation
            format_string = material.vertex_format
            formats = format_string.split("_")

            norm_index = None

            if VertexFormats.VERTS_3_FLOATS.value in formats:
                vert_index = formats.index(VertexFormats.VERTS_3_FLOATS.value)
            else:
                raise WavefrontError("Position Vertices not found in .obj file")
            if VertexFormats.NORMALS_3_FLOATS.value in formats:
                norm_index = formats.index(VertexFormats.NORMALS_3_FLOATS.value)

            # Contains the vertex list of floats in the format described above
            try:
                all_verts = np.array(material.vertices).reshape((int)(len(material.vertices)/3), 3)
            except ValueError as e:
                raise WavefrontError(e)

            vertices = all_verts[vert_index::len(formats)]
            indices = np.arange(len(vertices))
            model = Model(vertices, indices)

            if norm_index:
                normals = all_verts[vert_index::len(formats)]
                indices = np.array([indices, indices]).T
                model.indices = indices
                model.normals = normals

            models.append(model)

        return models


class VertexFormats(Enum):
    """Enum that declares the different types of .obj vertex declarations.

    values: T2F, C3F, N3F and V3F
    description:    T = Texture; C = Color; N = Normals; V = Vertex
                    2F = two floats, 3F = three floats

    """
    VERTS_3_FLOATS = "V3F"
    NORMALS_3_FLOATS = "N3F"
    UV_2_FLOATS = "T2F"
    COLOR_3_FLOATS = "C3F"







