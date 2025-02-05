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

class Scene:
    def __init__(self, shader, file_path):
        self.shader = shader
        self.vao = VAO()
        self.uma = UManager(self.shader)

        self.objects = []
        self.subObjs = []

        overall_min, overall_max = self.parse_file_pywavefront(file_path)

        self.min_x = overall_min['x']
        self.max_x = overall_max['x']
        self.min_y = overall_min['y']
        self.max_y = overall_max['y']
        self.min_z = overall_min['z']
        self.max_z = overall_max['z']
        self.dir_path = os.path.dirname(file_path)
        mtl_path = file_path.replace(".obj", ".mtl")
        self.materials = self.load_materials(mtl_path)
        self.split_obj()

    def parse_file_pywavefront(self, obj_file):
        scene = pywavefront.Wavefront(obj_file, collect_faces=False)

        overall_min = {'x': float('inf'), 'y': float('inf'), 'z': float('inf')}
        overall_max = {'x': float('-inf'), 'y': float('-inf'), 'z': float('-inf')}

        for mesh_name, mesh in scene.meshes.items():
            # print("Length of materials:", len(mesh.materials))
            if len(mesh.materials) == 1: # 1 object 1 material
                # print("each object has 1 materials")
                current_obj = {
                    'name': mesh_name,
                    'texcoords': [],
                    'normals': [],
                    'vertices': [],
                    'material': None,
                }
                # print("Has texture (sig)", mesh.materials[0].has_uvs)
                hasUV = mesh.materials[0].has_uvs
                self.process_material_data(mesh.materials[0], current_obj, hasUV, overall_min, overall_max)
                self.objects.append(current_obj)

            else: # 1 object many materials
                # print("has multiple materials")
                for mat in mesh.materials:
                    current_obj = {
                        'name': mat.name,
                        'texcoords': [],
                        'normals': [],
                        'vertices': [],
                        'material': None,
                    }
                    # print("Has texture (mul)", mat.has_uvs)
                    hasUV = mat.has_uvs
                    self.process_material_data(mat, current_obj, hasUV, overall_min, overall_max)
                    self.objects.append(current_obj)

        return overall_min, overall_max

    def process_material_data(self, material, current_obj, UV_flag, overall_min, overall_max):
        data = material.vertices  # [vt_x1, vt_y1, vn_x1    , vn_y1, vn_z1, v_x1, v_y1, v_z1 ...]
        current_obj['material'] = material.name

        # Define the size of each group
        num_texcoords = 2
        num_normals = 3
        num_vertices = 3

        if UV_flag:
            for i in range(0, len(data), num_texcoords + num_normals + num_vertices):
                current_obj['texcoords'].append(data[i:i + num_texcoords])
                current_obj['normals'].append(data[i + num_texcoords:i + num_texcoords + num_normals])

                vertex = data[i + num_texcoords + num_normals:i + num_texcoords + num_normals + num_vertices]
                current_obj['vertices'].append(vertex)

                # Update overall min and max
                overall_min['x'] = min(overall_min['x'], vertex[0])
                overall_min['y'] = min(overall_min['y'], vertex[1])
                overall_min['z'] = min(overall_min['z'], vertex[2])

                overall_max['x'] = max(overall_max['x'], vertex[0])
                overall_max['y'] = max(overall_max['y'], vertex[1])
                overall_max['z'] = max(overall_max['z'], vertex[2])
        else: # have no textcoords
            for i in range(0, len(data), num_normals + num_vertices):
                current_obj['normals'].append(data[i:i + num_normals])
                # Extract vertex coordinates
                vertex = data[i + num_normals:i + num_normals + num_vertices]
                current_obj['vertices'].append(vertex)

                # Update overall min and max
                overall_min['x'] = min(overall_min['x'], vertex[0])
                overall_min['y'] = min(overall_min['y'], vertex[1])
                overall_min['z'] = min(overall_min['z'], vertex[2])

                overall_max['x'] = max(overall_max['x'], vertex[0])
                overall_max['y'] = max(overall_max['y'], vertex[1])
                overall_max['z'] = max(overall_max['z'], vertex[2])

    def split_obj(self):
        print("length objects:", len(self.objects))
        for obj in self.objects:
            model = SubScene(self.shader,
                             obj['vertices'],
                             obj['texcoords'],
                             obj['normals'],
                             self.materials[obj['material']],
                             self.dir_path
                             ).setup()
            self.subObjs.append(model)

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