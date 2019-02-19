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

    # _generate_cube_example()
    _generate_tooth_example()


def _generate_tooth_example():
    # Define texture paths

    texture_path1 = prepend_dir('resources/images/texture_1_adjusted.bmp')
    texture_path4 = prepend_dir('resources/images/texture_4_adjusted.bmp')
    texture_path8 = prepend_dir('resources/images/texture_8_adjusted.bmp')
    texture_path11 = prepend_dir('resources/images/texture_11_adjusted.bmp')
    texture_path14 = prepend_dir('resources/images/texture_14_adjusted.bmp')
    texture_path19 = prepend_dir('resources/images/texture_19_adjusted.bmp')

    # Define output path
    output_path = '/tmp/'

    # Cam 1

    location = np.array([-1.22, 1.21, 9.8])
    rotation_euler = np.array([-8, 20.2, 85.2])
    rotation_quat = np.array([0.715, -0.169, 0.082, 0.674])
    focal_length = 35
    sensor_width = 32
    sensor_height = 25.6
    screen_width = 1280
    screen_height = 1024

    cam1 = Camera(focal_length, screen_width, screen_height, sensor_width, sensor_height, location, rotation_euler,
                  rotation_quat)

    # Cam 4

    location = np.array([9.8, 1.2, 1.22])
    rotation_euler = np.array([83.6, 7.29, 110])
    rotation_quat = np.array([0.461, 0.342, 0.572, 0.585])

    cam4 = Camera(focal_length, screen_width, screen_height, sensor_width, sensor_height, location, rotation_euler,
                  rotation_quat)

    # Cam 8

    location = np.array([-1.41, -0.64, -9.89])
    rotation_euler = np.array([187, -9.41, 86.6])
    rotation_quat = np.array([-0.103, 0.720, 0.686, 0.016])

    cam8 = Camera(focal_length, screen_width, screen_height, sensor_width, sensor_height, location, rotation_euler,
                  rotation_quat)

    # Cam 11

    location = np.array([-9.88, 0, -1.69])
    rotation_euler = np.array([259, 360, 76.4])
    rotation_quat = np.array([0.5, -0.606, -0.478, 0.392])

    cam11 = Camera(focal_length, screen_width, screen_height, sensor_width, sensor_height, location, rotation_euler,
                   rotation_quat)

    # Cam 14

    location = np.array([-2.47, 0.08, 9.77])
    rotation_euler = np.array([347, -347, 84.5])
    rotation_quat = np.array([0.721, -0.164, 0.007, 0.673])

    cam14 = Camera(focal_length, screen_width, screen_height, sensor_width, sensor_height, location, rotation_euler,
                   rotation_quat)

    # Cam 19

    location = np.array([2.39, 8.29, -4.38])
    rotation_euler = np.array([-60.3, 238, -36.1])
    rotation_quat = np.array([-0.265, 0.467, 0.642, 0.548])

    cam19 = Camera(focal_length, screen_width, screen_height, sensor_width, sensor_height, location, rotation_euler,
                   rotation_quat)

    # Create textures

    texture1 = Texture(texture_path1, cam1)
    texture4 = Texture(texture_path4, cam4)
    texture8 = Texture(texture_path8, cam8)
    texture11 = Texture(texture_path11, cam11)
    texture14 = Texture(texture_path14, cam14)
    texture19 = Texture(texture_path19, cam19)

    # Import Model

    print("Starting model import...")
    models = Wavefront.import_obj_as_model_list(prepend_dir('resources/models/4_downsized_adjusted.obj'))
    print("Finished model import...")

    model = models[0]

    # Create Mapper
    mapper = Mapper([texture1], model, 1280 // 8, 1024 // 8,
                    output_path, "Tooth")
    mapper.start_texture_mapping()


def _generate_cube_example():
    scipt_path = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))
    texture_path = scipt_path + '/resources/images/texture_cube.png'
    texture_path2 = scipt_path + '/resources/images/texture_cube_side.png'
    texture_path3 = scipt_path + '/resources/images/texture_cube_4.png'
    texture_path4 = scipt_path + '/resources/images/texture_cube_5.png'
    texture_path5 = scipt_path + '/resources/images/texture_cube_6.png'
    output_path = '/tmp/'

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

    cam2 = Camera(focal_length, screen_width, screen_height, sensor_width, sensor_height, location, rotation)

    # Create Cam3

    location = [4.28, 3.58, 0]
    rotation = [-90, 180, -52.2]

    cam3 = Camera(focal_length, screen_width, screen_height, sensor_width, sensor_height, location, rotation)

    # Create Cam4

    location = [4.28, 3.58, 2.91]
    rotation = [-119, 178, -47.8]

    cam4 = Camera(focal_length, screen_width, screen_height, sensor_width, sensor_height, location, rotation)

    # Create Cam5

    location = [5.45, -3.34, 2.91]
    rotation = [-116, 184, -118]

    cam5 = Camera(focal_length, screen_width, screen_height, sensor_width, sensor_height, location, rotation)

    # Create Cam6

    location = [-5, -5.14, 2.91]
    rotation = [-113, 180, -223]

    cam6 = Camera(focal_length, screen_width, screen_height, sensor_width, sensor_height, location, rotation)

    # Create textures

    texture1 = Texture(texture_path, cam1)
    texture2 = Texture(texture_path, cam2)
    texture3 = Texture(texture_path2, cam3)
    texture4 = Texture(texture_path3, cam4)
    texture5 = Texture(texture_path4, cam5)
    texture6 = Texture(texture_path5, cam6)

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
    mapper = Mapper([texture1, texture2, texture3, texture4, texture5, texture6], model, 96, 54, output_path, "Cube")
    mapper.start_texture_mapping()
