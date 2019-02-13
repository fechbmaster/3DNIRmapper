import numpy as np
from typing import List, Union

from nirmapper.model.colladaExporter import ColladaCreator
from nirmapper.renderer.renderer import Renderer
from nirmapper.renderer.texture import Texture
from nirmapper.model.model import Model


class Mapper(object):
    """The Mapper class is responsible for mapping coordinates from textures to the model.

    """

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
        print("Exporting textured model to: ", self.output_path)
        ColladaCreator.create_collada_from_model_with_textures(self.model, self.textures, self.output_path,
                                                               self.node_name)
        print("Finished - have a nice day!")

    def start_visibility_analysis(self):
        print("Starting visibility analysis...")
        for texture in self.textures:
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
        print("Finished visibility analysis...")
