import inspect
import os

import numpy as np

from nirmapper.mapper import Mapper
from nirmapper.model.colladaExporter import ColladaCreator
from nirmapper.model.model import Model
from nirmapper.model.wavefrontImporter import Wavefront, IndicesFormat
from nirmapper.renderer.renderer import Renderer
from nirmapper.renderer.texture import Texture, Camera


def prepend_dir(file):
    return os.path.join(os.path.dirname(__file__), file)


def main(argv=None):
    print("Welcome to 3DNIRMapper!")

    #_generate_cube_example()
    _generate_tooth_example()


def _generate_tooth_example():
    texture_path1 = prepend_dir('resources/images/texture_4_adjusted.bmp')
    texture_path2 = prepend_dir('resources/images/texture_11_adjusted.bmp')
    output_path = '/tmp/4_adjusted.dae'

    # Cam 1

    location = np.array([9.8, 1.2, 1.22])
    rotation_euler = np.array([83.6, 7.29, 110])
    rotation_quat = np.array([0.461, 0.342, 0.572, 0.585])
    focal_length = 35
    sensor_width = 32
    sensor_height = 25.6
    screen_width = 1280
    screen_height = 1024

    cam1 = Camera(focal_length, screen_width, screen_height, sensor_width, sensor_height, location, rotation_euler,
                  rotation_quat)

    # Cam 2

    location = np.array([-9.88, 0, -1.69])
    rotation_euler = np.array([259, 360, 76.4])
    rotation_quat = np.array([0.5, -0.606, -0.478, 0.392])
    focal_length = 35
    sensor_width = 32
    sensor_height = 25.6
    screen_width = 1280
    screen_height = 1024

    cam2 = Camera(focal_length, screen_width, screen_height, sensor_width, sensor_height, location, rotation_euler,
                  rotation_quat)

    # Create textures

    texture1 = Texture(texture_path1, cam1)
    texture2 = Texture(texture_path2, cam2)

    print("Starting model import...")
    models = Wavefront.import_obj_as_model_list(prepend_dir('resources/models/4_downsized_adjusted.obj'))
    print("Finished model import...")

    model = models[0]

    # Create Mapper
    mapper = Mapper([texture1, texture2], model, 1280, 10240, output_path, "Tooth")
    mapper.start_texture_mapping()


def _generate_cube_example():
    scipt_path = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))
    texture_path = scipt_path + '/resources/images/texture_cube.png'
    output_path = '/tmp/cube_example.dae'
    print("This will create a demo mapping of a cube in ", output_path, " using the renderer from: ", texture_path)

    # Create Cam1

    location = [0, 7, 0]
    rotation = [-90, 180, 0]
    focal_length = 35
    sensor_width = 32
    sensor_height = 18
    screen_width = 1920
    screen_height = 1080

    cam1 = Camera(focal_length, screen_width, screen_height, sensor_width, sensor_height, location, rotation)

    # Create Cam2

    location = [7, 0, 0]
    rotation = [-90, 180, -90]
    focal_length = 35
    sensor_width = 32
    sensor_height = 18
    screen_width = 1920
    screen_height = 1080

    cam2 = Camera(focal_length, screen_width, screen_height, sensor_width, sensor_height, location, rotation)

    # Create textures

    texture1 = Texture(texture_path, cam1)
    texture2 = Texture(texture_path, cam2)

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

    # Create Mapper
    mapper = Mapper([texture1, texture2], model, 40, 20, output_path, "Cube")
    mapper.start_texture_mapping()
