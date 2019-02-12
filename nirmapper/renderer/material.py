import string

from nirmapper.renderer.camera import Camera


class Material(object):
    visible_vertices = []
    vert_indices = []
    uv_coords = []
    uv_indices = []
    counts = []

    def __init__(self, texture_path: string, cam: Camera):
        self.texture_path = texture_path
        self.cam = cam
