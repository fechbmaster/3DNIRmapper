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
    texture_path = prepend_dir('resources/images/texture_4_adjusted.png')
    output_path = '/tmp/4_adjusted.dae'

    print("Welcome to 3DNIRMapper!")
    print("This will create a demo mapping of a cube in ", output_path, " using the texture from: ", texture_path)

    location = np.array([9.8, 1.2, 1.22])
    rotation = np.array([83.6, 7.29, 110])
    focal_length = 35
    sensor_width = 32
    sensor_height = 18
    screen_width = 1280
    screen_height = 1024

    scipt_path = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))

    cam = Camera(focal_length, screen_width, screen_height, sensor_width, sensor_height, location, rotation)

    print("Starting model import...")
    models: List[Model] = Wavefront.import_obj_as_model_list(prepend_dir('resources/models/4_downsized_adjusted.obj'))
    print("Finished model import...")

    model = models[0]

    # The magic is happening here
    uv_coords = cam.project_world_points_to_uv_coords(model.vertices)
    model.uv_coords = uv_coords

    ColladaCreator.create_collada_from_model(model, texture_path, output_path)
