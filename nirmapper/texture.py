import string

import numpy as np

from nirmapper import Camera, Model


class Texture(object):

    z_buffer: np.ndarray

    def __init__(self, text_id: int, texture_path: string, cam: Camera):
        self.id = text_id
        self.texture_path = texture_path
        self.cam = cam

    def check_occlusion_for_model(self, model: Model):
        return []

    def create_z_buffer(self, model: Model):
        width = self.cam.resolution_x
        height = self.cam.resolution_y

        z_buffer = np.zeros([width, height])

