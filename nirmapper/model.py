from collada import *
from collada import source
import numpy as np


class Model(object):
    def __init__(self, vertices, normals, uv_coords, indices):
        self.vertices = vertices
        self.normals = normals
        self.uv_coords = uv_coords
        self.indices = indices


class ColladaCreator(object):

    @staticmethod
    def create_collada(model: Model, texture_path, output_path):
        mesh = Collada()

        # needed for texture
        image = material.CImage("material_0-image", texture_path)
        surface = material.Surface("material_0-image-surface", image)
        sampler2d = material.Sampler2D("material_0-image-sampler", surface)
        map = material.Map(sampler2d, "UVSET0")

        effect = material.Effect("effect0", [surface, sampler2d], "lambert", emission=(0.0, 0.0, 0.0, 1),
                                 ambient=(0.0, 0.0, 0.0, 1), diffuse=map, transparent=map, transparency=0.0,
                                 double_sided=True)
        mat = material.Material("material0", "mymaterial", effect)

        mesh.effects.append(effect)
        mesh.materials.append(mat)
        mesh.images.append(image)

        vert_src = source.FloatSource("cubeverts-array",  np.array(model.vertices), ('X', 'Y', 'Z'))
        normal_src = source.FloatSource("cubenormals-array",  np.array(model.normals), ('X', 'Y', 'Z'))
        uv_src = source.FloatSource("cubeuv_array", np.array(model.uv_coords), ('S', 'T'))

        geom = geometry.Geometry(mesh, "geometry0", "mycube", [vert_src, normal_src, uv_src])
        input_list = source.InputList()
        input_list.addInput(0, 'VERTEX', "#cubeverts-array")
        input_list.addInput(1, 'TEXCOORD', "#cubeuv_array", set="0")
        input_list.addInput(2, 'NORMAL', "#cubenormals-array")

        triset = geom.createTriangleSet(model.indices, input_list, "materialref")
        geom.primitives.append(triset)
        mesh.geometries.append(geom)

        matnode = scene.MaterialNode("materialref", mat, inputs=[])
        geomnode = scene.GeometryNode(geom, [matnode])
        node = scene.Node("Cube", children=[geomnode])

        myscene = scene.Scene("myscene", [node])
        mesh.scenes.append(myscene)
        mesh.scene = myscene

        mesh.write(output_path)
