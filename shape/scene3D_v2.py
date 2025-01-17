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

class Scene:
    def __init__(self, shader, file_path):
        self.shader = shader
        self.vao = VAO()
        self.uma = UManager(self.shader)
        self.subobjs = []
        self.objects = self.parse_file_pywavefront(file_path)
        self.materials = self.load_materials(file_path)
        self.split_obj()

    def parse_file_pywavefront(self, obj_file):
        scene = pywavefront.Wavefront(obj_file, collect_faces=True)
        list_mesh = []
        # print("Length of mesh items:", len(scene.meshes.items()))

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
                self.process_material_data(mesh.materials[0], current_obj, hasUV)

                list_mesh.append(current_obj)

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
                    self.process_material_data(mat, current_obj, hasUV)
                    list_mesh.append(current_obj)

        return list_mesh

    def process_material_data(self, material, current_obj, UV_flag):
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
                current_obj['vertices'].append(
                    data[i + num_texcoords + num_normals:i + num_texcoords + num_normals + num_vertices])
        else:
            for i in range(0, len(data), num_normals + num_vertices):
                current_obj['normals'].append(data[i:i + num_normals])
                current_obj['vertices'].append(
                    data[i + num_normals:i + num_normals + num_vertices]
                )

    def load_materials(self, file_path):
        obj_file = pywavefront.Wavefront(file_path, create_materials=True)

        # Access materials
        self.materials = {}
        for name, material in obj_file.materials.items():
            def trim_list(lst):
                return lst[:-1] if isinstance(lst, list) and len(lst) == 4 else lst

            self.materials[name] = {
                'Ns': getattr(material, 'shininess', None),                                 # Specular exponent
                'Ni': getattr(material, 'optical_density', None),                           # Optical density
                'd': getattr(material, 'transparency', None),                               # Dissolve
                'Tr': 1 - getattr(material, 'transparency', 1.0),                           # Transparency (inverse dissolve)
                'Tf': getattr(material, 'transmission_filter', [1, 1, 1]),                  # Transmission filter
                'illum': getattr(material, 'illumination_model', None),                     # Illumination model
                'Ka': trim_list(getattr(material, 'ambient', [0, 0, 0])),                   # Ambient color
                'Kd': trim_list(getattr(material, 'diffuse', [0, 0, 0])),                   # Diffuse color
                'Ks': trim_list(getattr(material, 'specular', [0, 0, 0])),                  # Specular color
                'Ke': trim_list(getattr(material, 'emissive', [0, 0, 0])),                  # Emissive color
                'map_Ka': getattr(material, 'texture_ambient', None),                       # Ambient texture
                'map_Kd': getattr(material, 'texture', None),                               # Diffuse texture
                'map_bump': getattr(material, 'texture_bump', None)                         # Bump map
            }

        output_file = 'materials.txt'
        with open(output_file, 'w') as f:
            for material_name, properties in self.materials.items():
                f.write(f"Material: {material_name}\n")
                for key, value in properties.items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")

    def split_obj(self):
        for obj in self.objects:
            vertices = []
            tecos = []
            normals = []

            for sublist in obj['vert_obj_id']: # [[1 2 3][2 3 4 ]]
                for vert_id in sublist:
                    vertices.append(self.vertices[int(vert_id)])

            for sublist in obj['textcoords_obj_id']:
                for teco_id in sublist:
                    tecos.append(self.texcoords[int(teco_id)])

            for sublist in obj['normal_obj_id']:
                for normal_id in sublist:
                    normals.append(self.normals[int(normal_id)])

            # if obj['texture_name'] == 'wire_115115115':
            model = SubScene( self.shader,
                            vertices,
                            tecos,
                            normals,
                            self.materials[obj['texture_name']]
                            ).setup()
            self.subobjs.append(model)  
        return self
    
    def update_shader(self, shader):
        for subobj in self.subobjs:
            subobj.update_shader(shader)

    def update_colormap(self, selected_colormap):
        for subobj in self.subobjs:
            subobj.uma.upload_uniform_scalar1i(selected_colormap, 'colormap_selection')

    def update_near_far(self, near, far):
        for subobj in self.subobjs:
            subobj.uma.upload_uniform_scalar1f(near, 'near')
            subobj.uma.upload_uniform_scalar1f(far, 'far')

    def update_Kmat(self, diffuse, specular, ambient):
        for subobj in self.subobjs:
            subobj.update_Kmat(diffuse, specular, ambient)

    def setup(self):
        for subobj in self.subobjs:
            subobj.setup()

    def draw(self, model, view, projection, cameraPos):
        for subobj in self.subobjs:
            subobj.draw(model, view, projection, cameraPos)