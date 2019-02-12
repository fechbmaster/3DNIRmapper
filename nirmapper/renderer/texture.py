import string

from nirmapper.renderer.camera import Camera


class Texture(object):
    visible_vertices = []
    vert_indices = []
    uv_coords = []
    uv_indices = []
    counts = []
    vis_triangle_ids = []

    def __init__(self, texture_path: string, cam: Camera):
        self.texture_path = texture_path
        self.cam = cam
