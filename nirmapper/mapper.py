import numpy as np
from typing import List, Union

from nirmapper.model.colladaExporter import ColladaCreator
from nirmapper.renderer.renderer import Renderer
from nirmapper.renderer.texture import Texture
from nirmapper.model.model import Model


class Mapper(object):
    """The Mapper class is responsible for mapping coordinates from textures to the model.

    """
    all_ids = np.array([], dtype=int)
    duplicate_ids = np.array([], dtype=int)

    def __init__(self, textures: Union[List[Texture], Texture], model: Model, buffer_dim_width: int,
                 buffer_dim_height: int, output_path: str, node_name: str):
        self.textures = textures
        self.model = model
        self.renderer = Renderer()
        self.buffer_x = buffer_dim_width
        self.buffer_y = buffer_dim_height
        self.output_path = output_path
        self.node_name = node_name

        # Generate vertices sequence from describing indices
        vert_sequence = np.array(model.vertices[model.indices.flatten()])
        # Reshape the vert sequence to length/9x3x3 triangle Pairs
        self.triangles = vert_sequence.reshape(vert_sequence.size // 9, 3, 3)

    def start_texture_mapping(self):
        self.start_visibility_analysis()
        self.clean_duplicates()
        self.export_textured_model()

    def start_visibility_analysis(self):
        print("Starting visibility analysis...")
        tmp_ids = np.array([])
        for idx, texture in enumerate(self.textures):
            vis_vertices, ids, counts = self.renderer.get_visible_triangles(self.triangles, texture.cam,
                                                                            self.buffer_x, self.buffer_y)

            # Set visible vertices
            texture.visible_vertices = vis_vertices

            # Set vertex indices
            texture.verts_indices = self.model.indices[ids]

            # Set normal indices
            if np.size(self.model.normals) > 0:
                texture.normal_indices = self.model.normal_indices[ids]

            # Set uv coords
            uv_coords = texture.cam.get_texture_coords_for_vertices(vis_vertices)
            texture.uv_coords = uv_coords

            # Set uv indices -> these are just indices of the uv_coords array
            texture.uv_indices = np.arange(texture.uv_coords.size // 2)

            # Set counts
            texture.counts = counts

            # Set triangle ids
            texture.vis_triangle_ids = ids

            # Set multiple textured triangles
            tmp_ids = np.append(tmp_ids, ids)

        print("Finished visibility analysis...")
        # Set the list of all ids
        self.all_ids = tmp_ids

    def clean_duplicates(self):
        print("Cleaning up duplicates...")
        self.duplicate_ids = self.__get_duplicate_ids(self.all_ids)

        # Set the duplicates for the textures
        for texture in self.textures:
            dub_ids = np.array(np.where(np.in1d(self.duplicate_ids, texture.vis_triangle_ids)))
            dub_ids = dub_ids.flatten()
            texture.duplicate_triangle_ids = self.duplicate_ids[dub_ids]

        for id in self.duplicate_ids:
            best_text_idx = self.__get_best_texture_for_id(id)
            for idx, texture in enumerate(self.textures):
                if idx != best_text_idx:
                    # Remove the unwanted data
                    texture.remove_duplicate_data_at_triangle_id(id)
            self.duplicate_ids = np.delete(self.duplicate_ids, 0)

        print("Finished cleaning up duplicates...")

    def __get_best_texture_for_id(self, id: int):
        counts = []
        for texture in self.textures:
            index = np.where(texture.duplicate_triangle_ids == id)
            if np.size(index) == 0:
                counts.append(0)
            else:
                counts.append(texture.counts[index])
        return np.argmax(counts)

    def export_textured_model(self):
        print("Exporting textured model to: ", self.output_path)
        ColladaCreator.create_collada_from_model_with_textures(self.model, self.textures, self.output_path,
                                                               self.node_name)
        print("Finished - have a nice day!")

    @staticmethod
    def __get_duplicate_ids(id_list) -> np.ndarray:
        s = np.sort(np.array([id_list]), axis=None)
        return s[:-1][s[1:] == s[:-1]]
