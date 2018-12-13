import numpy as np
import os
import inspect
from .mapper import UVMapper
from .camera import Camera
from .model import Model, ColladaCreator


def main(argv=None):
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

    model = Model(verts, indices, normals, uv_coords)
    scipt_path = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))
    texture_path = scipt_path + '/resources/images/texture_cube.png'
    output_path = '/tmp/cube_example.dae'

    print("Welcome to 3DNIRMapper!")
    print("This will create a demo mapping of a cube in ", output_path, " using the texture from: ", texture_path)

    ColladaCreator.create_collada_from_model(model, texture_path, output_path)
