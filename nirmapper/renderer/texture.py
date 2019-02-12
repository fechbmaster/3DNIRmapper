import string

import numpy as np
from nirmapper.renderer.camera import Camera


class Texture(object):
    visible_triangle_ids = np.ndarray
    visible_triangle_counts = np.ndarray
    z_buffer: np.ndarray

    def __init__(self, texture_path: string, cam: Camera):
        self.texture_path = texture_path
        self.cam = cam
