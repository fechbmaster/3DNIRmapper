import numpy as np
import string

from nirmapper import Camera, Model


class Texture(object):

    z_buffer: np.ndarray

    def __init__(self, id: int, texture_path: string, cam: Camera):
        self.id = id
        self.texture_path = texture_path
        self.cam = cam

    def check_occlusion_for_model(self, model: Model):
        return []
