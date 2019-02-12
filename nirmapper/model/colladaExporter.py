from copy import copy
from typing import Tuple, List

import numpy as np
from collada import Collada, material, source, geometry, scene

from nirmapper.renderer.texture import Texture
from nirmapper.exceptions import ColladaError
from nirmapper.model.model import Model
from nirmapper.model.wavefrontImporter import IndicesFormat


class ColladaCreator(object):
    """A creator class for Collada files.
    """

    @staticmethod
    def create_collada_from_model_with_textures(model: Model, textures: List[Texture], output_path,
                                                node_name: str) -> None:
        """
        Create a Collada file out of an model and textures.
        :param model: The model.
        :param textures: Texture list
        :param output_path: Path where collada file should be stored.
        :param node_name: The name of the node, the object should carry later.
        """

        mesh = Collada()

        # Don't calculate faces for UV because not needed if textured with own textures
        faces, formats = ColladaCreator.generate_faces(model, ignore_uvs=True)

        # Vertices must be there so no need to check for that
        vert_src = source.FloatSource("cubeverts-array", np.array(model.vertices), ('X', 'Y', 'Z'))

        source_list = [vert_src]
        input_list = source.InputList()
        idx = 0
        input_list.addInput(idx, 'VERTEX', "#cubeverts-array")

        if IndicesFormat.N3F in formats:
            normal_src = source.FloatSource("cubenormals-array", np.array(model.normals), ('X', 'Y', 'Z'))
            source_list.append(normal_src)
            idx += 1
            input_list.addInput(idx, 'NORMAL', "#cubenormals-array")

        idx += 1
        input_list.addInput(idx, 'TEXCOORD', "#cubeuv_array", set="0")

        # Create material dict
        faces = faces.reshape(faces.size // np.size(formats), np.size(formats))
        geomnode_list = []
        for i, texture in enumerate(textures):
            mat, mat_id = ColladaCreator.insert_material_to_mesh(mesh, texture.texture_path, i)

            # check for uv coords in texture
            if (texture.uv_coords is None or texture.uv_coords.size == 0) and (
                    texture.uv_indices is None or texture.uv_indices.size == 0):
                raise ColladaError("UV indices or coordinates of texture are not defined!")

            # Set sources
            uv_src = source.FloatSource("cubeuv_array", np.array(texture.uv_coords), ('S', 'T'))
            tmp_source_list = copy(source_list)
            tmp_source_list.append(uv_src)

            # Set geoms
            geom = geometry.Geometry(mesh, "geometry%d" % i, "geometry%d" % i, tmp_source_list)

            # Extend faces
            uv_indices = texture.uv_indices.reshape(texture.uv_indices.size, 1)
            tmp_faces = np.hstack([faces, uv_indices]).astype(int)

            # Set triset
            triset = geom.createTriangleSet(tmp_faces, input_list, mat_id)
            geom.primitives.append(triset)
            mesh.geometries.append(geom)

            # Set matnode
            matnode = scene.MaterialNode(mat_id, mat, inputs=[])

            # Set geomnode
            geomnode = scene.GeometryNode(geom, [matnode])
            geomnode_list.append(geomnode)

        # this is just for debugging and actually not needed
        # formats.append(IndicesFormat.T2F)

        node = scene.Node(node_name, children=geomnode_list)

        # pycollada rotates the model for some reason - prevent that
        rotation1 = scene.RotateTransform(1.0, 0.0, 0.0, -90)
        node.transforms.append(rotation1)

        myscene = scene.Scene("scene0", [node])
        mesh.scenes.append(myscene)
        mesh.scene = myscene

        mesh.write(output_path)

    @staticmethod
    def insert_material_to_mesh(mesh: Collada, texture_path: str, id: int) -> Tuple[material.Material, str]:
        # needed for renderer
        image = material.CImage("material_%d-image" % id, texture_path)
        surface = material.Surface("material_%d-image-surface" % id, image)
        sampler2d = material.Sampler2D("material_%d-image-sampler" % id, surface)
        mat_map = material.Map(sampler2d, "UVSET0")

        effect = material.Effect("effect0", [surface, sampler2d], "lambert", emission=(0.0, 0.0, 0.0, 1),
                                 ambient=(0.0, 0.0, 0.0, 1), diffuse=mat_map, transparent=mat_map, transparency=0.0,
                                 double_sided=True)
        mat = material.Material("material_%d_ID" % id, "material_%d" % id, effect)

        mesh.effects.append(effect)
        mesh.materials.append(mat)
        mesh.images.append(image)

        return mat, "material_%d" % id

    @staticmethod
    def insert_default_material_to_mesh(mesh: Collada) -> Tuple[material.Material, str]:
        effect = material.Effect("effect0", [], "phong", diffuse=(1, 0, 0), specular=(0, 1, 0))
        mat = material.Material("material0_ID", "material_0", effect)
        mesh.effects.append(effect)
        mesh.materials.append(mat)

        return mat, "material_0"

    @staticmethod
    def create_collada_from_model(model: Model, output_path: str, node_name: str, texture_path: str = None) -> None:
        """
        Create a Collada file out of an model.

        :param model: The model.
        :param texture_path: Path of the texture.
        :param output_path: Path where collada file should be stored.
        :param node_name: The name of the node, the object should carry later.
        """
        mesh = Collada()

        faces, formats = ColladaCreator.generate_faces(model)

        # Vertices must be there so no need to check for that
        vert_src = source.FloatSource("cubeverts-array", np.array(model.vertices), ('X', 'Y', 'Z'))

        geometry_list = [vert_src]
        input_list = source.InputList()
        idx = 0
        input_list.addInput(idx, 'VERTEX', "#cubeverts-array")

        if IndicesFormat.N3F in formats:
            normal_src = source.FloatSource("cubenormals-array", np.array(model.normals), ('X', 'Y', 'Z'))
            geometry_list.append(normal_src)
            idx += 1
            input_list.addInput(idx, 'NORMAL', "#cubenormals-array")
        if IndicesFormat.T2F in formats:
            uv_src = source.FloatSource("cubeuv_array", np.array(model.uv_coords), ('S', 'T'))
            geometry_list.append(uv_src)
            mat, mat_identifier = ColladaCreator.insert_material_to_mesh(mesh, texture_path, 0)
            idx += 1
            input_list.addInput(idx, 'TEXCOORD', "#cubeuv_array", set="0")
        else:
            mat, mat_identifier = ColladaCreator.insert_default_material_to_mesh(mesh)

        geom = geometry.Geometry(mesh, "geometry0", "geometry0", geometry_list)
        triset = geom.createTriangleSet(faces, input_list, mat_identifier)
        geom.primitives.append(triset)
        mesh.geometries.append(geom)

        matnode = scene.MaterialNode(mat_identifier, mat, inputs=[])
        geomnode = scene.GeometryNode(geom, [matnode])
        node = scene.Node(node_name, children=[geomnode])

        # pycollada rotates the model for some reason - prevent that
        rotation1 = scene.RotateTransform(1.0, 0.0, 0.0, -90)
        node.transforms.append(rotation1)

        myscene = scene.Scene("scene0", [node])
        mesh.scenes.append(myscene)
        mesh.scene = myscene

        mesh.write(output_path)

    @staticmethod
    def generate_faces(model: Model, ignore_uvs: bool = False) -> Tuple[np.ndarray, List[IndicesFormat]]:
        """
        Method generates faces out of the given model data
        :param ignore_uvs: Indicates if uv coordinates should be ignored. Needed for multiple texturing.
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
        if ignore_uvs is not True:
            if model.uv_coords.size > 0:
                if model.uv_indices is None or model.uv_indices.size == 0:
                    raise ColladaError("UV indices are not defined!")
                uv_indices = model.uv_indices.reshape(model.uv_indices.size, 1)
                faces = faces.reshape(faces.size // np.size(formats), np.size(formats))
                faces = np.hstack([faces, uv_indices])
                formats.append(IndicesFormat.T2F)
        return faces.astype(int), formats
