import inspect
import os

import numpy as np

from nirmapper.model.colladaExporter import ColladaCreator
from nirmapper.model.model import Model
from nirmapper.model.wavefrontImporter import Wavefront, IndicesFormat
from nirmapper.renderer.renderer import Renderer
from nirmapper.renderer.texture import Texture, Camera
from .mapper import UVMapper


def prepend_dir(file):
    return os.path.join(os.path.dirname(__file__), file)


def main(argv=None):
    print("Welcome to 3DNIRMapper!")

    # _generate_cube_example()
    _generate_tooth_example()


def _generate_tooth_example():
    texture_path = prepend_dir('resources/images/texture_4_adjusted.bmp')
    output_path = '/tmp/4_adjusted.dae'
    print("This will create a demo mapping of a cube in ", output_path, " using the renderer from: ", texture_path)

    location = np.array([9.8, 1.2, 1.22])
    rotation_euler = np.array([83.6, 7.29, 110])
    rotation_quat = np.array([0.461, 0.342, 0.572, 0.585])
    focal_length = 35
    sensor_width = 32
    sensor_height = 25.6
    screen_width = 1280
    screen_height = 1024

    cam = Camera(focal_length, screen_width, screen_height, sensor_width, sensor_height, location, rotation_euler,
                 rotation_quat)

    print("Starting model import...")
    models = Wavefront.import_obj_as_model_list(prepend_dir('resources/models/4_downsized_adjusted.obj'))
    print("Finished model import...")

    model = models[0]

    print("Starting texturing...")

    # Check visible verts
    vis_vertices, ids, counts = \
        Renderer.get_visible_triangles(model.vertices,
                                       model.indices,
                                       cam, 1280, 1024)

    uv_coords = cam.get_texture_coords_for_vertices(vis_vertices)
    model.uv_coords = uv_coords

    vis_tri_indices = np.arange(0, vis_vertices.size // 3)

    uv_indices = np.zeros(model.indices.shape)
    uv_indices[ids] = vis_tri_indices.reshape(np.size(ids), 3)

    model.uv_indices = uv_indices

    print("Finished texturing...")

    ColladaCreator.create_collada_from_model(model, texture_path, output_path, "Tooth_4")


def _generate_cube_example():
    scipt_path = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))
    texture_path = scipt_path + '/resources/images/texture_cube.png'
    output_path = '/tmp/cube_example.dae'
    print("This will create a demo mapping of a cube in ", output_path, " using the renderer from: ", texture_path)

    # Create Cam

    location = np.array([0, 7, 0])
    rotation = np.array([-90, 180, 0])
    focal_length = 35
    sensor_width = 32
    sensor_height = 18
    screen_width = 1920
    screen_height = 1080

    cam = Camera(focal_length, screen_width, screen_height, sensor_width, sensor_height, location, rotation)

    # Create model

    verts = np.array([[1, 1, -1],  # 1
                      [1, -1, -1],  # 2
                      [-1, -1, -1],  # 3
                      [-1, 1, -1],  # 4
                      [1, 1, 1],  # 5
                      [1, -1, 1],  # 6
                      [-1, -1, 1],  # 7
                      [-1, 1, 1]])  # 8

    normals = np.array([0, 0, -1,
                        0, 0, 1,
                        1, 0, 0,
                        0, -1, 0,
                        -1, 0, 0,
                        0, 1, 0,
                        0, 0, -1,
                        0, 0, 1,
                        1, 0, 0,
                        0, -1, 0,
                        -1, 0, 0,
                        0, 1, 0])

    indices = np.array([0,
                        2,
                        3,
                        7,
                        5,
                        4,
                        4,
                        1,
                        0,
                        5,
                        2,
                        1,
                        2,
                        7,
                        3,
                        0,
                        7,
                        4,
                        0,
                        1,
                        2,
                        7,
                        6,
                        5,
                        4,
                        5,
                        1,
                        5,
                        6,
                        2,
                        2,
                        6,
                        7,
                        0,
                        3,
                        7])

    normal_indices = indices

    model = Model(verts, normals)
    model.indices = indices
    model.normal_indices = normal_indices

    print("Starting texturing...")

    # Check visible verts
    vis_vertices, ids, counts = \
        Renderer.get_visible_triangles(model.vertices,
                                       model.indices,
                                       cam, 40, 20)

    uv_coords = cam.get_texture_coords_for_vertices(vis_vertices)
    model.uv_coords = uv_coords

    vis_tri_indices = np.arange(0, vis_vertices.size // 3)

    uv_indices = np.zeros(model.indices.shape)
    uv_indices[ids] = vis_tri_indices.reshape(np.size(ids), 3)

    model.uv_indices = uv_indices

    print("Finished texturing...")

    # Calculate UVs

    ColladaCreator.create_collada_from_model(model, texture_path, output_path, "Cube")
