import ntpath
import os
import shutil
from typing import Tuple, List

import numpy as np
from collada import Collada, material, source, geometry, scene, asset

from nirmapper.data.model import Model
from nirmapper.data.texture import Texture
from nirmapper.exceptions import ColladaError
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
        axis = asset.UP_AXIS.Z_UP
        mesh.assetInfo.upaxis = axis

        # Define data for 3d model

        vertices = model.vertices
        normals = model.normals

        for idx, texture in enumerate(textures):
            if texture.visible_vertices is None or np.size(texture.visible_vertices) == 0:
                textures = np.delete(textures, idx)

        # Generate data structure
        output_path = ColladaCreator.__create_subfolder_at_output_path(output_path, node_name + 'MappedCollada')
        ColladaCreator.__copy_textures_output_path(textures, output_path)

        combined_uvs = np.array([])
        for texture in textures:
            start_index = np.size(combined_uvs) // 2
            texture.arange_uv_indices(start_index)
            combined_uvs = np.append(combined_uvs, texture.uv_coords)

        # === Define sources ===
        source_list = [source.FloatSource("verts-array", np.array(vertices), ('X', 'Y', 'Z'))]
        # normals
        if np.size(normals) > 0:
            source_list.append(source.FloatSource("normals-array", np.array(normals), ('X', 'Y', 'Z')))
        source_list.append(source.FloatSource("uv_array", np.array(combined_uvs), ('S', 'T')))

        # Add plain model to see uncolored triangles
        # plain_geomnode = ColladaCreator.get_plain_model_geomnode(mesh, model, source_list, 0)

        geomnode_list = []
        # geomnode_list.append(plain_geomnode)
        matnode_list = []

        # Set geom for textured verts
        geom = geometry.Geometry(mesh, "geometry0", "geometry0", source_list)

        for i, texture in enumerate(textures):
            mat, mat_id = ColladaCreator.insert_texture_material_to_mesh(mesh, ntpath.basename(texture.texture_path), i)

            # Set input list
            input_list = source.InputList()
            j = 0
            input_list.addInput(j, 'VERTEX', "#verts-array")
            j += 1
            if np.size(normals) > 0:
                input_list.addInput(j, 'NORMAL', "#normals-array")
                j += 1
            input_list.addInput(j, 'TEXCOORD', "#uv_array", set="0")

            # Extend faces
            text_faces = texture.verts_indices.reshape(texture.verts_indices.size, 1)
            if np.size(normals) > 0:
                text_faces = np.hstack([text_faces, texture.normal_indices.reshape(texture.normal_indices.size, 1)])
            text_faces = np.hstack([text_faces, texture.uv_indices.reshape(texture.uv_indices.size, 1)]).astype(int)

            # Set triset
            triset = geom.createTriangleSet(text_faces, input_list, mat_id)
            geom.primitives.append(triset)

            # Set matnode
            matnode = scene.MaterialNode(mat_id, mat, inputs=[])
            matnode_list.append(matnode)

        # Set geomnode
        mesh.geometries.append(geom)
        geomnode = scene.GeometryNode(geom, matnode_list)
        geomnode_list.append(geomnode)

        ColladaCreator.write_out_geomnodes(mesh, geomnode_list, output_path, node_name)

    @staticmethod
    def get_plain_model_geomnode(mesh: Collada, model: Model, source_list: List[source.FloatSource],
                                 mat_id: int) -> scene.GeometryNode:
        """
        Method returns a plain model with the default material. Can be used to combine the texture geoms with this plain one.
        :param Collada mesh: The Collada mesh.
        :param Model model: The model.
        :param List[source.FloatSource] source_list: The list with the FloatSource.
        :param int mat_id: The id of the material.
        :return scene.GeometryNode: The geomnode of the plain model.
        """
        # Create plain material
        plain_mat, plain_mat_id = ColladaCreator.insert_plain_material_to_mesh(mesh, mat_id)

        # Create plain input list

        # === Define input list for plain model ===
        plain_input_list = source.InputList()
        plain_input_list.addInput(0, 'VERTEX', "#verts-array")
        if np.size(model.normals) > 0:
            plain_input_list.addInput(1, 'NORMAL', "#normals-array")

        # === Define plain geometry ===
        plain_geom = geometry.Geometry(mesh, "mesh1-geometry", "mesh1-geometry", source_list)

        # === Define plain faces ===
        faces, formats = ColladaCreator.__generate_faces(model, True)

        # === Create plain triset ===
        plain_triset = plain_geom.createTriangleSet(faces, plain_input_list, plain_mat_id)

        # === Combine to geomnode
        plain_geom.primitives.append(plain_triset)
        mesh.geometries.append(plain_geom)
        plain_matnode = scene.MaterialNode(plain_mat_id, plain_mat, inputs=[])
        plain_geomnode = scene.GeometryNode(plain_geom, [plain_matnode])

        return plain_geomnode

    @staticmethod
    def insert_texture_material_to_mesh(mesh: Collada, texture_path: str, id: int) -> Tuple[material.Material, str]:
        """
        Method inserts a texture as material to the Collada mesh.
        :param Collada mesh: The Collada mesh.
        :param str texture_path: The path of the texture.
        :param int id: The material id.
        :return Tuple[material.Material, str]: Returns a material.Material and the identifier as string.
        """
        # needed for renderer
        image = material.CImage("material_%d-image" % id, texture_path)
        surface = material.Surface("material_%d-image-surface" % id, image)
        sampler2d = material.Sampler2D("material_%d-image-sampler" % id, surface)
        mat_map = material.Map(sampler2d, "UVSET0")

        effect = material.Effect("effect%d" % id, [surface, sampler2d], "lambert", emission=(0.0, 0.0, 0.0, 1),
                                 ambient=(0.0, 0.0, 0.0, 1), diffuse=mat_map, transparent=mat_map, transparency=0.0,
                                 double_sided=True)
        mat = material.Material("material_%d_ID" % id, "material_%d" % id, effect)

        mesh.effects.append(effect)
        mesh.materials.append(mat)
        mesh.images.append(image)

        return mat, "material_%d" % id

    @staticmethod
    def insert_plain_material_to_mesh(mesh: Collada, id: int) -> Tuple[material.Material, str]:
        """
        Method inserts a plain material to the Collada mesh.
        :param Collada mesh: The Collada mesh.
        :param int id: The material id.
        :return Tuple[material.Material, str]: Returns a material.Material and the identifier as string.
        """
        effect = material.Effect("effect%d" % id, [], "phong", diffuse=(0, 0, 0), specular=(0, 0, 0))
        mat = material.Material("material_%d_ID" % id, "material_%d" % id, effect)
        mesh.effects.append(effect)
        mesh.materials.append(mat)

        return mat, "material_%d" % id

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
        axis = asset.UP_AXIS.Z_UP
        mesh.assetInfo.upaxis = axis

        faces, formats = ColladaCreator.__generate_faces(model)

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
            mat, mat_identifier = ColladaCreator.insert_texture_material_to_mesh(mesh, texture_path, 0)
            idx += 1
            input_list.addInput(idx, 'TEXCOORD', "#cubeuv_array", set="0")
        else:
            mat, mat_identifier = ColladaCreator.insert_plain_material_to_mesh(mesh, 0)

        geom = geometry.Geometry(mesh, "geometry0", "geometry0", geometry_list)
        triset = geom.createTriangleSet(faces, input_list, mat_identifier)
        geom.primitives.append(triset)
        mesh.geometries.append(geom)

        matnode = scene.MaterialNode(mat_identifier, mat, inputs=[])
        geomnode = scene.GeometryNode(geom, [matnode])

        ColladaCreator.write_out_geomnodes(mesh, [geomnode], output_path, node_name)

    @staticmethod
    def write_out_geomnodes(mesh: Collada, geomnodes: List[scene.GeometryNode], output_path: str,
                            node_name: str) -> None:
        """
        Method writes out geomnodes to an ouput path.
        :param Collada mesh: The Collada mesh.
        :param  List[scene.GeometryNode] geomnodes:
        :param str output_path: The output path.
        :param str node_name: The name of the node.
        """
        node = scene.Node(node_name, children=geomnodes)

        myscene = scene.Scene("scene0", [node])
        mesh.scenes.append(myscene)
        mesh.scene = myscene

        mesh.write(output_path + node_name + ".dae")

    @staticmethod
    def __copy_textures_output_path(textures: List[Texture], output_path: str) -> None:
        """
        Method copys textures to an output path.
        :param List[Texture] textures: The tetures to copy.
        :param output_path: The output path.
        :return None
        """
        for texture in textures:
            file_name = ntpath.basename(texture.texture_path)
            shutil.copy2(texture.texture_path, output_path + file_name)

    @staticmethod
    def __create_subfolder_at_output_path(output_path: str, folder_name: str) -> str:
        """
        Method creates a subfolder at output path.
        :param str output_path: The path where the subfolder should be created.
        :param str folder_name: The name of the subfoolder
        :return str: The combined subfolder path.
        """
        compbined_path = output_path + folder_name
        if not os.path.exists(compbined_path):
            os.makedirs(compbined_path)
        return compbined_path + '/'

    @staticmethod
    def __generate_faces(model: Model, ignore_uvs: bool = False) -> Tuple[np.ndarray, List[IndicesFormat]]:
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
