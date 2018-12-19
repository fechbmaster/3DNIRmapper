from typing import List

import numpy as np
import os
import inspect
from .mapper import UVMapper
from .camera import Camera
from .model import Model, ColladaCreator, Wavefront


def prepend_dir(file):
    return os.path.join(os.path.dirname(__file__), file)


def main(argv=None):
    print("Welcome to 3DNIRMapper!")

    #_generate_cube_example()
    _generate_tooth_example()


def _generate_tooth_example():
    texture_path = prepend_dir('resources/images/texture_4_adjusted.bmp')
    output_path = '/tmp/4_adjusted.dae'
    print("This will create a demo mapping of a cube in ", output_path, " using the texture from: ", texture_path)

    location = np.array([9.8, 1.2, 1.22])
    rotation = np.array([83.6, 7.29, 110])
    focal_length = 35
    sensor_width = 32
    sensor_height = 32
    screen_width = 1280
    screen_height = 1024

    scipt_path = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))

    cam = Camera(focal_length, screen_width, screen_height, sensor_width, sensor_height, location, rotation)

    print("Starting model import...")
    models: List[Model] = Wavefront.import_obj_as_model_list(prepend_dir('resources/models/4_downsized_adjusted.obj'))
    print("Finished model import...")

    model = models[0]

    # The magic is happening here
    uv_coords = cam.project_world_points_to_uv_coords(model.obj_vertices)
    model.uv_coords = uv_coords

    # Update indices
    indices, ind_format = model.generate_indices()
    model.set_indices(indices, ind_format)

    ColladaCreator.create_collada_from_model(model, texture_path, output_path, "Tooth_4")


def _generate_cube_example():
    location = np.array([0, 7, 0])
    rotation = np.array([-90, 180, 0])
    focal_length = 35
    sensor_width = 32
    sensor_height = 18
    screen_width = 1920
    screen_height = 1080

    cam = Camera(focal_length, screen_width, screen_height, sensor_width, sensor_height, location, rotation)

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

    indices = np.array([0, 0, 0,
                        2, 2, 0,
                        3, 3, 0,
                        7, 7, 1,
                        5, 5, 1,
                        4, 4, 1,
                        4, 4, 2,
                        1, 1, 2,
                        0, 0, 2,
                        5, 5, 3,
                        2, 2, 3,
                        1, 1, 3,
                        2, 2, 4,
                        7, 7, 4,
                        3, 3, 4,
                        0, 0, 5,
                        7, 7, 5,
                        4, 4, 5,
                        0, 0, 6,
                        1, 1, 6,
                        2, 2, 6,
                        7, 7, 7,
                        6, 6, 7,
                        5, 5, 7,
                        4, 4, 8,
                        5, 5, 8,
                        1, 1, 8,
                        5, 5, 9,
                        6, 6, 9,
                        2, 2, 9,
                        2, 2, 10,
                        6, 6, 10,
                        7, 7, 10,
                        0, 0, 11,
                        3, 3, 11,
                        7, 7, 11])

    # The magic is happening here
    uv_coords = cam.project_world_points_to_uv_coords(verts)

    model = Model(verts, normals, uv_coords)
    model.set_indices(indices, "V3F_N3F_T2F")
    scipt_path = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))
    texture_path = scipt_path + '/resources/images/texture_cube.png'
    output_path = '/tmp/cube_example.dae'

    print("Welcome to 3DNIRMapper!")
    print("This will create a demo mapping of a cube in ", output_path, " using the texture from: ", texture_path)

    ColladaCreator.create_collada_from_model(model, texture_path, output_path, "Cube")
