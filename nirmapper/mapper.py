import multiprocessing
import time
from typing import List, Union

import numpy as np

from nirmapper.data.model import Model
from nirmapper.data.texture import Texture
from nirmapper.model.colladaExporter import ColladaCreator
from nirmapper.renderer.renderer import Renderer
from nirmapper.utils import generate_triangle_sequence


class Mapper(object):
    """The Mapper class is responsible for mapping coordinates from textures to the model.

    """
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

        # Reshape the vert sequence to length/9x3x3 triangle Pairs
        self.triangles = generate_triangle_sequence(model.vertices, model.indices)

    def start_texture_mapping(self, mutli_threaded=True):
        print("Starting visibility analysis...")
        start = time.time()
        self.start_visibility_analysis(mutli_threaded)
        end = time.time()
        duration = end - start
        print("Finished visibility analysis. Time exceeded: ", duration)
        print("Cleaning up duplicates...")
        start = time.time()
        self.clean_duplicates()
        end = time.time()
        duration = end - start
        print("Finished cleaning up duplicates. Time exceeded: ", duration)
        print("Exporting textured model to: ", self.output_path)
        self.export_textured_model()
        print("Finished - have a nice day!")

    def start_visibility_analysis(self, multi_threaded=True):
        tmp_ids = np.array([], dtype=int)
        if multi_threaded:
            manager = multiprocessing.Manager()
            return_dict = manager.dict()
            thread_list = []
            for idx, texture in enumerate(self.textures):
                p = multiprocessing.Process(target=self.__start_parallel_visibility_analysis_for_texture,
                                            args=(texture, idx, return_dict))
                thread_list.append(p)
            for p in thread_list:
                p.start()
            for p in thread_list:
                p.join()

            for idx, texture in enumerate(self.textures):
                self.textures[idx] = return_dict.get(idx)
        else:
            for texture in self.textures:
                self.start_visibility_analysis_for_texture(texture)

        for texture in self.textures:
            # Set multiple textured triangles
            tmp_ids = np.append(tmp_ids, texture.vis_triangle_indices)

        # Set the list of all ids
        self.__set_duplicate_ids(tmp_ids)

    def __start_parallel_visibility_analysis_for_texture(self, texture: Texture, i: int, return_dict):
        result = self.start_visibility_analysis_for_texture(texture)
        return_dict[i] = result

    def start_visibility_analysis_for_texture(self, texture: Texture) -> Texture:
        vis_vertices, ids, counts = \
            self.renderer.get_visible_triangles(self.model.vertices, self.model.indices, texture.cam,
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
        texture.arange_uv_indices()

        # Set counts
        texture.counts = counts

        # Set triangle ids
        texture.vis_triangle_indices = ids

        return texture

    def clean_duplicates(self):
        self.set_duplicates_for_textures()

        for id in self.duplicate_ids:
            best_texture_idx = self.get_best_texture_for_duplicate_triangle(id)
            for idx, texture in enumerate(self.textures):
                if idx == best_texture_idx:
                    texture.remove_duplicate_with_index(id)
                else:
                    # Remove the unwanted data
                    texture.remove_triangle_with_index(id)

        self.duplicate_ids = []

    def set_duplicates_for_textures(self):
        # Set the duplicates for the textures
        for texture in self.textures:
            dub_ids = np.array(np.where(np.in1d(self.duplicate_ids, texture.vis_triangle_indices)), dtype=int)
            dub_ids = dub_ids.flatten()
            texture.duplicate_triangle_indices = self.duplicate_ids[dub_ids]

    def get_best_texture_for_duplicate_triangle(self, triangle_index: int):
        counts = []
        for texture in self.textures:
            try:
                index = list(texture.vis_triangle_indices).index(triangle_index)
                counts.append(texture.counts[index])
            except ValueError:
                counts.append(0)

        # Return first texture id with best count
        return np.argmax(counts)

    def export_textured_model(self):
        ColladaCreator.create_collada_from_model_with_textures(self.model, self.textures, self.output_path,
                                                               self.node_name)

    def __set_duplicate_ids(self, id_list):
        s = np.sort(np.array([id_list]), axis=None)
        s = np.array(s[:-1][s[1:] == s[:-1]], dtype=int)

        self.duplicate_ids = np.unique(s)
