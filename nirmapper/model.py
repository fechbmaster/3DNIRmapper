import ntpath
import shutil
import string
from enum import Enum
from typing import List, Tuple

import numpy as np
import pywavefront
from collada import Collada, asset, source, geometry, scene, material

from nirmapper.camera import Camera
from nirmapper.exceptions import ModelError, ColladaError, WavefrontError, TextureError
from nirmapper.utils import quaternion_matrix


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


class Texture(object):
    __visible_vertices = []
    __vert_indices = []
    __uv_coords = []
    __uv_indices = np.array([], dtype=int)
    __normal_indices = []
    counts = np.array([], dtype=int)
    vis_triangle_indices = np.array([], dtype=int)
    duplicate_triangle_indices = np.array([], dtype=int)

    def __init__(self, texture_path: string, cam: Camera):
        self.texture_path = texture_path
        self.cam = cam

    @property
    def visible_vertices(self) -> np.ndarray:
        return self.__visible_vertices

    @visible_vertices.setter
    def visible_vertices(self, visible_vertices: np.ndarray):
        if visible_vertices is None or np.size(visible_vertices) == 0:
            self.__visible_vertices = []
            return
        visible_vertices = self.__reshape(visible_vertices, 3)
        self.__visible_vertices = visible_vertices

    @property
    def verts_indices(self) -> np.ndarray:
        return self.__vert_indices

    @verts_indices.setter
    def verts_indices(self, verts_indices: np.ndarray):
        if verts_indices is None or np.size(verts_indices) == 0:
            self.__vert_indices = []
            return
        visible_vertices = self.__reshape(verts_indices, 3)
        self.__vert_indices = visible_vertices

    @property
    def uv_coords(self) -> np.ndarray:
        return self.__uv_coords

    @uv_coords.setter
    def uv_coords(self, uv_coords: np.ndarray):
        if uv_coords is None or np.size(uv_coords) == 0:
            self.__uv_coords = []
            return
        # Reshape to get coords
        uv_coords = self.__reshape(uv_coords, 2)
        self.__uv_coords = uv_coords

    @property
    def uv_indices(self) -> np.ndarray:
        return self.__uv_indices

    def arange_uv_indices(self, start_index: int = 0):
        self.__uv_indices = np.arange(start_index, start_index + (np.size(self.uv_coords) // 2))

    @property
    def normal_indices(self) -> np.ndarray:
        return self.__normal_indices

    @normal_indices.setter
    def normal_indices(self, normal_indices: np.ndarray):
        if normal_indices is None or np.size(normal_indices) == 0:
            self.__normal_indices = []
            return
        # Reshape to get coords
        normal_indices = self.__reshape(normal_indices, 3)
        self.__normal_indices = normal_indices

    def remove_triangle_with_index(self, triangle_index: int):
        """
        Removes triangle data if triangle_index is in visible triangle indices list.
        :param int triangle_index: The index of the triangle.
        :return None:
        """
        if triangle_index in self.vis_triangle_indices:
            # Delete elements
            idx = list(self.vis_triangle_indices).index(triangle_index)
            self.__delete_triangle_at_index(idx)
            self.remove_duplicate_with_index(triangle_index)

    def remove_duplicate_with_index(self, triangle_index: int):
        """
        Removes the duplciate triangle at triangle_index.
        :param triangle_index: The index to remove.
        :return None:
        """
        if triangle_index in self.duplicate_triangle_indices:
            # Delete duplicate if there
            idx = list(self.duplicate_triangle_indices).index(triangle_index)
            self.duplicate_triangle_indices = np.delete(self.duplicate_triangle_indices, idx)

    def __delete_triangle_at_index(self, tri_idx: int):
        # Delete verts
        indices = np.arange(tri_idx * 3, (tri_idx + 1) * 3)
        self.visible_vertices = np.delete(self.visible_vertices, indices, axis=0)
        # Delete vert indices
        self.verts_indices = np.delete(self.verts_indices, tri_idx, axis=0)
        # Delete uvs
        self.uv_coords = np.delete(self.uv_coords, indices, axis=0)
        # Re-arange uv indices
        self.arange_uv_indices()
        # Delete normal indices
        self.normal_indices = np.delete(self.normal_indices, tri_idx, axis=0)
        # Delete triangle id
        self.vis_triangle_indices = np.delete(self.vis_triangle_indices, tri_idx)
        # Delete counts
        self.counts = np.delete(self.counts, tri_idx)

    @staticmethod
    def __reshape(array, vert_length: int) -> np.ndarray:
        """
        Method reshapes an array depending on its length by giving the vertices length.

        :param array: The array that should be reshaped.
        :param vert_length: The length of the vectors.
        :return: The reshaped array.
        """
        try:
            array = np.array(array)
            return array.reshape([(array.size // vert_length), vert_length])
        except ValueError as e:
            raise TextureError(e)


class IndicesFormat(Enum):
    """Enum that holds the valid formats for the indices.

    """
    T2F = 1,
    # C3F = 2,   # not needed here
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
        Method returns a plain model with the default material.
        Can be used to combine the texture geoms with this plain one.
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
    def insert_texture_material_to_mesh(mesh: Collada, texture_path: str, mat_id: int) -> Tuple[material.Material, str]:
        """
        Method inserts a texture as material to the Collada mesh.
        :param Collada mesh: The Collada mesh.
        :param str texture_path: The path of the texture.
        :param int mat_id: The material id.
        :return Tuple[material.Material, str]: Returns a material.Material and the identifier as string.
        """
        # needed for renderer
        image = material.CImage("material_%d-image" % mat_id, texture_path)
        surface = material.Surface("material_%d-image-surface" % mat_id, image)
        sampler2d = material.Sampler2D("material_%d-image-sampler" % mat_id, surface)
        mat_map = material.Map(sampler2d, "UVSET0")

        effect = material.Effect("effect%d" % mat_id, [surface, sampler2d], "lambert", emission=(0.0, 0.0, 0.0, 1),
                                 ambient=(0.0, 0.0, 0.0, 1), diffuse=mat_map, double_sided=True)
        mat = material.Material("material_%d_ID" % mat_id, "material_%d" % mat_id, effect)

        mesh.effects.append(effect)
        mesh.materials.append(mat)
        mesh.images.append(image)

        return mat, "material_%d" % mat_id

    @staticmethod
    def insert_plain_material_to_mesh(mesh: Collada, mat_id: int) -> Tuple[material.Material, str]:
        """
        Method inserts a plain material to the Collada mesh.
        :param Collada mesh: The Collada mesh.
        :param int mat_id: The material id.
        :return Tuple[material.Material, str]: Returns a material.Material and the identifier as string.
        """
        effect = material.Effect("effect%d" % mat_id, [], "phong", diffuse=(0, 0, 0), specular=(0, 0, 0))
        mat = material.Material("material_%d_ID" % mat_id, "material_%d" % mat_id, effect)
        mesh.effects.append(effect)
        mesh.materials.append(mat)

        return mat, "material_%d" % mat_id

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
        """
        Returns the index of a given format.
        :param ind_format: The format
        :return: The index
        """
        return self.formats.index(ind_format)

    @staticmethod
    def generate_indices(length: int) -> np.ndarray:
        """
        Generates indices depending on the given vertices.

        :return: An array of indices, The format of the indices.
        """
        return np.arange(0, length)
