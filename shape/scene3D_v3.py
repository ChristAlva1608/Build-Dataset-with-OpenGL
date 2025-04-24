import random

import glfw
import numpy as np
from OpenGL.GL import *
import ctypes
from PIL import Image
import pywavefront
from libs.shader import *
from libs.transform import *
from libs.buffer import *
from libs.camera import *
import glm
from itertools import cycle
from shape.subScene import *
import os
import copy


class Scene:
    def __init__(self, shader, file_path, scene_net_flag, nyu_rgb_paths):
        self.shader = shader
        # self.uma = UManager(self.shader)
        self.subObjs = []

        self.dir_path = os.path.dirname(file_path)
        self.name = os.path.basename(file_path)[:-4]
        self.obj_names_list = []

        self.sceneNet_flag = scene_net_flag
        self.NYU_rgb_paths = nyu_rgb_paths

        self.parse_file_pywavefront(file_path)

    def parse_file_pywavefront(self, obj_file):
        scene = pywavefront.Wavefront(obj_file, collect_faces=True, parse=True, create_materials=True)
        print("Finished Parsing PyWavefront")

        if scene.mtllibs:
            print(f"Material libraries: {scene.mtllibs}")
            mtl_path = os.path.join(self.dir_path, scene.mtllibs[0])
        else:
            mtl_path = obj_file.replace(".obj", ".mtl")
        self.materials = self.load_materials(mtl_path)

        mat_dict ={}

        for mesh_name, mesh in scene.meshes.items():
            self.obj_names_list.append(mesh_name)
            num_faces = len(mesh.faces) # auto correct for triang and quad
            mat = mesh.materials[0] # access to material
            mat_name = mat.name
            if mat_name not in mat_dict:
                mat_dict[mat_name] = 0

            texcoords, normals, vertices = self.mat_data(mat, mat_dict[mat_name], mat_dict[mat_name] + num_faces*3)
            mat_dict[mat_name] = mat_dict[mat_name] + num_faces*3

            NYU_path = random.choice(self.NYU_rgb_paths) if self.NYU_rgb_paths else None
            model = SubScene(
                self.shader,
                vertices,
                texcoords,
                normals,
                self.materials.get(mat_name, None),
                self.dir_path,
                self.sceneNet_flag,
                NYU_path
            ).setup()

            self.subObjs.append(model)
        print("Finish parsing meshes")

    def mat_data(self, material, start, end):
        data = material.vertices  # List of vertex data
        num_texcoords = 2
        num_normals = 3
        num_vertices = 3

        texcoords = []
        normals = []
        vertices = []

        if material.has_uvs:
            for i in range(start*8, end*8, num_texcoords + num_normals + num_vertices):
                texcoords.extend(data[i:i + num_texcoords])
                normals.extend(data[i + num_texcoords:i + num_texcoords + num_normals])
                vertex = data[i + num_texcoords + num_normals:i + num_texcoords + num_normals + num_vertices]
                vertices.extend(vertex)
        else:
            for i in range(start*6, end*6, num_normals + num_vertices):
                normals.extend(data[i:i + num_normals])
                vertex = data[i + num_normals:i + num_normals + num_vertices]
                vertices.extend(vertex)

        return texcoords, normals, vertices

    def process_material_data(self, material):
        data = material.vertices  # List of vertex data
        num_texcoords = 2
        num_normals = 3
        num_vertices = 3

        texcoords = []
        normals = []
        vertices = []

        if material.has_uvs:
            for i in range(0, len(data), num_texcoords + num_normals + num_vertices):
                texcoords.extend(data[i:i + num_texcoords])
                normals.extend(data[i + num_texcoords:i + num_texcoords + num_normals])
                vertex = data[i + num_texcoords + num_normals:i + num_texcoords + num_normals + num_vertices]
                vertices.extend(vertex)
        else:
            for i in range(0, len(data), num_normals + num_vertices):
                normals.extend(data[i:i + num_normals])
                vertex = data[i + num_normals:i + num_normals + num_vertices]
                vertices.extend(vertex)

        return texcoords, normals, vertices

    def load_materials(self, file_path):
        materials = {}
        current_material = None

        def parse_vector(values):
            # Convert space-separated string of numbers into list of floats
            return [float(x) for x in values.strip().split()]

        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                parts = line.strip().split(maxsplit=1)
                if len(parts) < 2:
                    continue

                keyword, value = parts

                if keyword == 'newmtl':
                    current_material = value
                    materials[current_material] = {
                        'Ns': None,  # Specular exponent
                        'Ni': None,  # Optical density
                        'd': None,  # Dissolve
                        'Tr': None,  # Transparency
                        'Tf': None,  # Transmission filter
                        'illum': None,  # Illumination model
                        'Ka': None,  # Ambient color
                        'Kd': None,  # Diffuse color
                        'Ks': None,  # Specular color
                        'Ke': None,  # Emissive color
                        'map_Ka': None,  # Ambient texture
                        'map_Kd': None,  # Diffuse texture
                        'map_bump': None  # Bump map
                    }

                elif current_material is not None:
                    mat = materials[current_material]

                    if keyword == 'Ns':
                        mat['Ns'] = float(value)
                    elif keyword == 'Ni':
                        mat['Ni'] = float(value)
                    elif keyword == 'd':
                        mat['d'] = float(value)
                    elif keyword == 'Tr':
                        mat['Tr'] = float(value)
                    elif keyword == 'Tf':
                        mat['Tf'] = parse_vector(value)
                    elif keyword == 'illum':
                        mat['illum'] = int(value)
                    elif keyword == 'Ka':
                        mat['Ka'] = parse_vector(value)
                    elif keyword == 'Kd':
                        mat['Kd'] = parse_vector(value)
                    elif keyword == 'Ks':
                        mat['Ks'] = parse_vector(value)
                    elif keyword == 'Ke':
                        mat['Ke'] = parse_vector(value)
                    elif keyword == 'map_Ka':
                        # Handle potential bump map parameters
                        parts = value.split()
                        mat['map_Ka'] = parts[-1]
                    elif keyword == 'map_Kd':
                        parts = value.split()
                        mat['map_Kd'] = parts[-1]
                    elif keyword == 'map_bump':
                        # Handle bump map with potential -bm parameter
                        parts = value.split()
                        if '-bm' in parts:
                            bm_index = parts.index('-bm')
                            # Store both the bump multiplier and filename
                            mat['map_bump'] = {
                                'multiplier': float(parts[bm_index + 1]),
                                'filename': parts[-1]
                            }
                        else:
                            mat['map_bump'] = {'filename': parts[-1], 'multiplier': 1.0}
        return materials

    def set_mode(self, num):
        for subobj in self.subObjs:
            subobj.uma.upload_uniform_scalar1i(num, 'mode')

    def update_colormap(self, selected_colormap):
        for subobj in self.subObjs:
            subobj.uma.upload_uniform_scalar1i(selected_colormap, 'colormap_selection')

    def update_near_far(self, near, far):
        for subobj in self.subObjs:
            subobj.uma.upload_uniform_scalar1f(near, 'near')
            subobj.uma.upload_uniform_scalar1f(far, 'far')

    def update_lightPos(self, lightPos):
        for subobj in self.subObjs:
            subobj.update_lightPos(lightPos)

    def update_lightColor(self, lightColor):
        for subobj in self.subObjs:
            subobj.update_lightColor(lightColor)

    def update_shininess(self, shininess):
        for subobj in self.subObjs:
            subobj.update_shininess(shininess)

    def update_attribute(self, attr, value):
        update_name = 'update_' + attr
        for subobj in self.subObjs:
            if hasattr(subobj, update_name):
                method = getattr(subobj, update_name)
                method(value)

    def setup(self):
        for subobj in self.subObjs:
            subobj.setup()

    def draw(self, cameraPos):
        for subobj in self.subObjs:
            subobj.draw(cameraPos)

    def get_model_matrix(self):
        return self.subObjs[0].get_model_matrix()

    def change_NYU_texture(self):
        # perform change all the scene texture to new NYU image
        for subScene in self.subObjs:
            NYU_path = random.choice(self.NYU_rgb_paths) if self.NYU_rgb_paths else None
            subScene.update_texture(NYU_path)

    def update_colors(self, color_list):
        # print(f"Color list length: {len(color_list)}") # 93
        for i, subobj in enumerate(self.subObjs): # still correct
            # print(f"Update object {self.obj_names_list[i]} with color {color_list[i]}")
            subobj.update_colors(color_list[i])

