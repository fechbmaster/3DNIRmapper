import inspect
import os

import numpy as np

from nirmapper.camera import Camera
from nirmapper.model import Model, Wavefront, Texture
from nirmapper.nirmapper import Mapper


def prepend_dir(file):
    return os.path.join(os.path.dirname(__file__), file)


def generate_tooth_example(src):
    # Define texture paths

    texture_path1 = prepend_dir('resources/images/texture_1_adjusted.bmp')
    texture_path4 = prepend_dir('resources/images/texture_4_adjusted.bmp')
    texture_path8 = prepend_dir('resources/images/texture_8_adjusted.bmp')
    texture_path11 = prepend_dir('resources/images/texture_11_adjusted.bmp')
    texture_path14 = prepend_dir('resources/images/texture_14_adjusted.bmp')
    texture_path19 = prepend_dir('resources/images/texture_19_adjusted.bmp')

    # Define output path
    output_path = src

    # Cam 1

    location = np.array([-1.22, 1.21, 9.8])
    rotation_quat = np.array([0.715, -0.169, 0.082, 0.674])
    focal_length = 35
    sensor_width = 32
    sensor_height = 25.6
    screen_width = 1280
    screen_height = 1024

    cam1 = Camera(focal_length, screen_width, screen_height, sensor_width, sensor_height, location, rotation_quat)

    # Cam 4

    location = np.array([9.8, 1.2, 1.22])
    rotation_quat = np.array([0.461, 0.342, 0.572, 0.585])

    cam4 = Camera(focal_length, screen_width, screen_height, sensor_width, sensor_height, location, rotation_quat)

    # Cam 8

    location = np.array([-1.41, -0.64, -9.89])
    rotation_quat = np.array([-0.103, 0.720, 0.686, 0.016])

    cam8 = Camera(focal_length, screen_width, screen_height, sensor_width, sensor_height, location, rotation_quat)

    # Cam 11

    location = np.array([-9.88, 0, -1.69])
    rotation_quat = np.array([0.5, -0.606, -0.478, 0.392])

    cam11 = Camera(focal_length, screen_width, screen_height, sensor_width, sensor_height, location, rotation_quat)

    # Cam 14

    location = np.array([-2.47, 0.08, 9.77])
    rotation_quat = np.array([0.721, -0.164, 0.007, 0.673])

    cam14 = Camera(focal_length, screen_width, screen_height, sensor_width, sensor_height, location, rotation_quat)

    # Cam 19

    location = np.array([2.39, 8.29, -4.38])
    rotation_quat = np.array([-0.265, 0.467, 0.642, 0.548])

    cam19 = Camera(focal_length, screen_width, screen_height, sensor_width, sensor_height, location, rotation_quat)

    # Create textures

    texture1 = Texture(texture_path1, cam1)
    texture4 = Texture(texture_path4, cam4)
    texture8 = Texture(texture_path8, cam8)
    texture11 = Texture(texture_path11, cam11)
    texture14 = Texture(texture_path14, cam14)
    texture19 = Texture(texture_path19, cam19)

    # Import Model

    print("Starting model import...")
    models = Wavefront.import_obj_as_model_list(prepend_dir('resources/models/4_downsized_adjusted_retriangulated.obj'))
    print("Finished model import...")

    model = models[0]

    # Create Mapper
    texture_mapper = Mapper([texture1, texture4, texture8, texture11, texture19], model, output_path, "Tooth", 0.5)
    texture_mapper.start_texture_mapping()


def generate_elephant_example(src):
    # Define texture paths

    texture_path0 = prepend_dir('resources/images/elefant_0.png')
    texture_path1 = prepend_dir('resources/images/elefant_1.png')
    texture_path2 = prepend_dir('resources/images/elefant_2.png')
    texture_path3 = prepend_dir('resources/images/elefant_3.png')
    texture_path4 = prepend_dir('resources/images/elefant_4.png')

    # Define output path
    output_path = src

    # Cam 0

    location = np.array([7.2106, -6.4976, 5.3436])
    rotation_quat = np.array([0.755, 0.441, 0.219, 0.437])
    focal_length = 35
    sensor_width = 32
    sensor_height = 25.6
    screen_width = 1024
    screen_height = 768

    cam0 = Camera(focal_length, screen_width, screen_height, sensor_width, sensor_height, location, rotation_quat)

    # Cam 1

    location = np.array([-2.3297, -12.285, 2.0324])
    rotation_quat = np.array([0.776, 0.628, -0.018, -0.050])

    cam1 = Camera(focal_length, screen_width, screen_height, sensor_width, sensor_height, location, rotation_quat)

    # Cam 2

    location = np.array([-7.4062, -6.7939, 4.9206])
    rotation_quat = np.array([0.804, 0.444, -0.214, -0.332])

    cam2 = Camera(focal_length, screen_width, screen_height, sensor_width, sensor_height, location, rotation_quat)

    # Cam 3

    location = np.array([1.8513, -6.7936, 9.9944])
    rotation_quat = np.array([0.956, 0.255, 0.114, 0.086])

    cam3 = Camera(focal_length, screen_width, screen_height, sensor_width, sensor_height, location, rotation_quat)

    # Cam 4

    location = np.array([-2.3530, 12.018, 1.7597])
    rotation_quat = np.array([0.063, 0.073, -0.624, -0.775])

    cam4 = Camera(focal_length, screen_width, screen_height, sensor_width, sensor_height, location, rotation_quat)

    # Create textures

    texture0 = Texture(texture_path0, cam0)
    texture1 = Texture(texture_path1, cam1)
    texture2 = Texture(texture_path2, cam2)
    texture3 = Texture(texture_path3, cam3)
    texture4 = Texture(texture_path4, cam4)
    # Import Model

    print("Starting model import...")
    models = Wavefront.import_obj_as_model_list(prepend_dir('resources/models/elefante.obj'))
    print("Finished model import...")

    model = models[0]

    # Create Mapper
    texture_mapper = Mapper([texture0, texture1, texture2, texture3, texture4], model,
                            output_path, "Elephant", 0.5)
    texture_mapper.start_texture_mapping()


def generate_cube_example(src):
    scipt_path = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))
    texture_path1 = scipt_path + '/resources/images/cube_1.png'
    texture_path2 = scipt_path + '/resources/images/cube_2.png'
    texture_path3 = scipt_path + '/resources/images/cube_3.png'
    texture_path4 = scipt_path + '/resources/images/cube_4.png'
    texture_path5 = scipt_path + '/resources/images/cube_5.png'
    texture_path6 = scipt_path + '/resources/images/cube_6.png'
    output_path = src

    # Create Cam1

    location = [0, 7, 0]
    rotation = [0.0, 0.0, 0.707, 0.707]
    focal_length = 35
    sensor_width = 32
    sensor_height = 18
    screen_width = 1920
    screen_height = 1080

    cam1 = Camera(focal_length, screen_width, screen_height, sensor_width, sensor_height, location, rotation)

    # Create Cam2

    location = [7.0, 0.0, 0.0]
    rotation = [0.5, 0.5, 0.5, 0.5]

    cam2 = Camera(focal_length, screen_width, screen_height, sensor_width, sensor_height, location, rotation)

    # Create Cam3

    location = [0.0, -7.0, 0]
    rotation = [0.707, 0.707, 0.0, 0.0]

    cam3 = Camera(focal_length, screen_width, screen_height, sensor_width, sensor_height, location, rotation)

    # Create Cam4

    location = [-7.0, 0.0, 0.0]
    rotation = [0.5, 0.5, -0.5, -0.5]

    cam4 = Camera(focal_length, screen_width, screen_height, sensor_width, sensor_height, location, rotation)

    # Create Cam5

    location = [0.0, 0.0, 7.0]
    rotation = [-0.707, 0.0, 0.0, 0.707]

    cam5 = Camera(focal_length, screen_width, screen_height, sensor_width, sensor_height, location, rotation)

    # Create Cam6

    location = [0.0, 0.0, -7.0]
    rotation = [0.0, 0.707, 0.707, -0.0]

    cam6 = Camera(focal_length, screen_width, screen_height, sensor_width, sensor_height, location, rotation)

    # Create textures

    texture1 = Texture(texture_path1, cam1)
    texture2 = Texture(texture_path2, cam2)
    texture3 = Texture(texture_path3, cam3)
    texture4 = Texture(texture_path4, cam4)
    texture5 = Texture(texture_path5, cam5)
    texture6 = Texture(texture_path6, cam6)

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
    texture_mapper = Mapper([texture1, texture2, texture3, texture4, texture5, texture6], model, output_path, "Cube",
                            0.05)
    texture_mapper.start_texture_mapping()
