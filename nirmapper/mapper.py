import numpy as np
from typing import List, Union

from nirmapper.renderer.renderer import Renderer
from nirmapper.renderer.texture import Texture
from nirmapper.model.model import Model


class Mapper(object):
    """The Mapper class is responsible for mapping coordinates from textures to the model.

    """

    def __init__(self, textures: Union[List[Texture], Texture], model: Model, buffer_dim_width: int, buffer_dim_height: int):
        self.textures = textures
        self.model = model
        self.renderer = Renderer()
        self.buffer_x = buffer_dim_width
        self.buffer_y = buffer_dim_height

        # Generate vertices sequence from describing indices
        vert_sequence = np.array(model.vertices[model.indices.flatten()])
        # Reshape the vert sequence to length/9x3x3 triangle Pairs
        self.triangles = vert_sequence.reshape(vert_sequence.size // 9, 3, 3)

    def start_texture_mapping(self):
        self.start_visibility_analysis()

    def start_visibility_analysis(self):
        print("Starting visibility analysis...")
        for texture in self.textures:
            vis_vertices, ids, counts = self.renderer.get_visible_triangles(self.triangles, texture.cam,
                                                                            self.buffer_x, self.buffer_y)

            # Set visible vertices
            texture.visible_vertices = vis_vertices

            # Set uv coords
            uv_coords = texture.cam.get_texture_coords_for_vertices(vis_vertices)
            texture.uv_coords = uv_coords

            # Set uv indices
            vis_tri_indices = np.arange(0, vis_vertices.size // 3)
            uv_indices = np.zeros(self.model.indices.shape)
            uv_indices[ids] = vis_tri_indices.reshape(np.size(ids), 3)
            texture.uv_indices = uv_indices

            # Set counts
            texture.counts = counts

            # Set triangle ids
            texture.vis_triangle_ids = ids
        print("Finished visibility analysis...")
