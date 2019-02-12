from typing import Tuple, List

import numpy as np
from collada import Collada, material, source, geometry, scene

from nirmapper.exceptions import ColladaError
from nirmapper.model.model import Model
from nirmapper.model.wavefrontImporter import IndicesFormat


class ColladaCreator(object):
    """A creator class for Collada files.
    """

    @staticmethod
    def create_collada_from_model(model: Model, texture_path: str, output_path: str, node_name: str) -> None:
        """
        Create a Collada file out of an modell and a renderer.

        :param model: The model.
        :param texture_path: Path of the renderer.
        :param output_path: Path where collada file should be stored.
        :param node_name: The name of the node, the object should carry later.
        """
        mesh = Collada()

        # needed for renderer
        image = material.CImage("material_0-image", texture_path)
        surface = material.Surface("material_0-image-surface", image)
        sampler2d = material.Sampler2D("material_0-image-sampler", surface)
        mat_map = material.Map(sampler2d, "UVSET0")

        effect = material.Effect("effect0", [surface, sampler2d], "lambert", emission=(0.0, 0.0, 0.0, 1),
                                 ambient=(0.0, 0.0, 0.0, 1), diffuse=mat_map, transparent=mat_map, transparency=0.0,
                                 double_sided=True)
        mat = material.Material("material0", "mymaterial", effect)

        mesh.effects.append(effect)
        mesh.materials.append(mat)
        mesh.images.append(image)

        faces, formats = ColladaCreator.generate_faces(model)

        # Vertices must be there so no need to check for that
        vert_src = source.FloatSource("cubeverts-array", np.array(model.vertices), ('X', 'Y', 'Z'))
        geometry_list = [vert_src]
        if IndicesFormat.N3F in formats:
            normal_src = source.FloatSource("cubenormals-array", np.array(model.normals), ('X', 'Y', 'Z'))
            geometry_list.append(normal_src)
        if IndicesFormat.T2F in formats:
            uv_src = source.FloatSource("cubeuv_array", np.array(model.uv_coords), ('S', 'T'))
            geometry_list.append(uv_src)

        geom = geometry.Geometry(mesh, "geometry0", "mycube", geometry_list)
        input_list = source.InputList()
        input_list.addInput(0, 'VERTEX', "#cubeverts-array")
        input_list.addInput(1, 'NORMAL', "#cubenormals-array")
        input_list.addInput(2, 'TEXCOORD', "#cubeuv_array", set="0")

        triset = geom.createTriangleSet(faces, input_list, "materialref")
        geom.primitives.append(triset)
        mesh.geometries.append(geom)

        matnode = scene.MaterialNode("materialref", mat, inputs=[])
        geomnode = scene.GeometryNode(geom, [matnode])
        node = scene.Node(node_name, children=[geomnode])

        # pycollada rotates the model for some reason - prevent that
        rotation1 = scene.RotateTransform(1.0, 0.0, 0.0, -90)
        node.transforms.append(rotation1)

        myscene = scene.Scene("myscene", [node])
        mesh.scenes.append(myscene)
        mesh.scene = myscene

        mesh.write(output_path)

    @staticmethod
    def generate_faces(model: Model) -> Tuple[np.ndarray, List[IndicesFormat]]:
        """
        Method generates faces out of the given model data
        :param Model model: The model
        :return Tuple[np.ndarray, List[IndicesFormat]]: Returns the faces and the format of the faces
        """
        if model.vertices is None or model.vertices.size == 0:
            raise ColladaError("Vertices are not defined!")
        if model.indices is None or model.indices.size == 0:
            raise ColladaError("Indices are not defined!")
        faces = model.indices.reshape(model.indices.size, 1)
        formats = [IndicesFormat.V3F]
        if model.normals.size > 0:
            if model.normal_indices is None or model.normal_indices.size == 0:
                raise ColladaError("Normal indices are not defined!")
            normal_indices = model.normal_indices.reshape(model.normal_indices.size, 1)
            faces = np.hstack([faces, normal_indices])
            formats.append(IndicesFormat.N3F)
        if model.uv_coords.size > 0:
            if model.uv_indices is None or model.uv_indices.size == 0:
                raise ColladaError("UV indices are not defined!")
            uv_indices = model.uv_indices.reshape(model.uv_indices.size, 1)
            faces = faces.reshape(faces.size // np.size(formats), np.size(formats))
            faces = np.hstack([faces, uv_indices])
            formats.append(IndicesFormat.T2F)

        return faces.astype(int), formats
