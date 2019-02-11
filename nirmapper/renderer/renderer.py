import numpy as np

from nirmapper.renderer.camera import Camera


class Renderer(object):

    @staticmethod
    def get_visible_triangles(vertices: np.ndarray, indices: np.ndarray, cam: Camera, buffer_size_x: int,
                              buffer_size_y: int):
        # Generate vertices sequence from describing indices
        vert_sequence = np.array(vertices[indices.flatten()])

        # Reshape the vert sequence to length/9x3x3 triangle Pairs
        triangles = vert_sequence.reshape(vert_sequence.size // 9, 3, 3)

        render_cam = \
            Camera(cam.focal_length_in_mm, buffer_size_x, buffer_size_y, cam.sensor_width_in_mm,
                            cam.sensor_height_in_mm, cam.cam_location_xyz, cam.cam_euler_rotation,
                            cam.cam_quat_rotation)
        z_buffer = Renderer.create_z_buffer(triangles, render_cam)
        ind_indices = np.array(z_buffer[:, :, 0]).astype(int)
        ind_indices, counts = np.unique(ind_indices[ind_indices > -1], return_counts=True)
        vis_vertices = triangles[ind_indices]
        vis_vertices = vis_vertices.reshape(vis_vertices.size // 3, 3)
        return vis_vertices, ind_indices, counts

    @staticmethod
    def create_z_buffer(triangles: np.ndarray, render_camera: Camera):
        buffer_width = render_camera.resolution_x - 1
        buffer_height = render_camera.resolution_y - 1

        z_buffer = np.full([buffer_width, buffer_height, 2], [-1, np.inf])

        for idx, triangle in enumerate(triangles):
            included_pixels = Renderer.rasterize(triangle, render_camera)
            for pixel in included_pixels:
                uvz_coords = render_camera.get_pixel_coords_for_vertices(triangle, include_z_value=True)
                # todo: evaluate this
                # mean is ok here because we don't have to check triangles that get cut by others
                z_value = np.mean(uvz_coords[:, -1:])
                # todo: could lead to 'z-fighting'
                if z_value < z_buffer[pixel[0], pixel[1]][1]:
                    z_buffer[pixel[0], pixel[1]] = [idx, z_value]

        #print(z_buffer[:, :, 0])
        return z_buffer

    @staticmethod
    def rasterize(vertices: np.ndarray, renderer_cam: Camera) -> np.ndarray:
        if vertices.shape != (3, 3):
            raise ValueError("Given triangle must be of shape (3, 3).")

        # Get renderer coords for vertice
        text_coords = renderer_cam.get_pixel_coords_for_vertices(vertices)
        bounding_box = Renderer.get_bounding_box_coords_for_triangle(text_coords)

        included_pixels = []
        for pixel in bounding_box:
            if Renderer.barycentric(pixel, text_coords):
                included_pixels.append(pixel)

        return np.array(included_pixels, dtype=int)

    # def pixel_is_included_in_triangle(self, text_coords: np.ndarray, pixel) -> bool:
    #     if text_coords.shape != (3, 2):
    #         raise ValueError("Triangle is in wrong shape. Shape must be (3, 2).")
    #     if len(pixel) != 2:
    #         raise ValueError("Pixel must have two coordinates.")
    #
    #     # first sort triangle coords by their x value
    #     coords = self.__sort_triangles_by_x_coord(text_coords)
    #
    #     inside = True
    #
    #     inside &= self.__edge_function(coords[0], coords[1], pixel)
    #     inside &= self.__edge_function(coords[1], coords[2], pixel)
    #     inside &= self.__edge_function(coords[2], coords[0], pixel)
    #
    #     return inside

    @staticmethod
    def get_bounding_box_coords_for_triangle(text_coords: np.ndarray) -> np.ndarray:
        min_x = np.amin(text_coords[:, 0])
        max_x = np.amax(text_coords[:, 0])
        min_y = np.amin(text_coords[:, 1])
        max_y = np.amax(text_coords[:, 1])

        x = np.arange(min_x, max_x + 1)
        y = np.arange(min_y, max_y + 1)

        box = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)

        return box

    # @staticmethod
    # def __sort_triangles_by_x_coord(triangles) -> np.ndarray:
    #     index = np.lexsort((triangles[:, 1], triangles[:, 0]))
    #     return triangles[index]

    @staticmethod
    def barycentric(p, text_coords: np.ndarray):
        v0, v1, v2 = text_coords[1] - text_coords[0], text_coords[2] - text_coords[0], p - text_coords[0]
        den = v0[0] * v1[1] - v1[0] * v0[1]
        if den == 0:
            return False
        v = (v2[0] * v1[1] - v1[0] * v2[1]) / den
        w = (v0[0] * v2[1] - v2[0] * v0[1]) / den
        u = 1.0 - v - w
        # Make near zero values to zero
        if np.isclose([u], [0])[0]:
            u = 0

        return (u >= 0) and (v >= 0) and (u + v <= 1)

    @staticmethod
    def __edge_function(v1, v2, p) -> bool:
        """
        The edge function determines if a point p is right, left or on line of a edge
        defined by two renderer coordinates v1 and v2.

        E(P) > 0 if P is to the "right" side
        E(P) = 0 if P is exactly on the line
        E(P) < 0 if P is to the "left " side

        :param v1: first renderer coordinate
        :param v2: second renderer coordinate
        :param p: point to check
        :return bool: returns true if point is on or right of the edge
        """
        return (p[0] - v1[0]) * (v2[1] - v1[1]) - (p[1] - v1[1]) * (v2[0] - v1[0]) >= 0
