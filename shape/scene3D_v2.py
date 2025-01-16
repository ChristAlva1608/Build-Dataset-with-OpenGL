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
        self.vertices, self.texcoords, self.normals, self.objects = self.parse_obj_file(file_path)
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

    def process_material_data(material, current_obj, UV_flag):
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
                
    def parse_obj_file(self,file_path):
        vertices_all = []
        normals_all = []
        texcoords_all = []

        # read the first time for vertices and texcoords only
        with open(file_path, 'r') as f:  # Use 'r' for text files
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):  # Skip empty lines and comments
                    continue

                parts = line.split()
                if parts[0] == 'v':  # Vertex definition
                    # Parse the vertex coordinates (x, y, z)
                    vertices_all.append([float(parts[1]), float(parts[2]), float(parts[3])])

                if parts[0] == 'vn':  # Normal definition
                    # Parse the normal coordinates (x, y, z)
                    normals_all.append([float(parts[1]), float(parts[2]), float(parts[3])])

                elif parts[0] == 'vt':  # Texture coordinate definition
                    # Parse the texture coordinates (u, v, [w])
                    u = float(parts[1])
                    v = float(parts[2])
                    w = float(parts[3]) if len(parts) > 3 else 0.0  # Default w to 0.0 if not provided
                    texcoords_all.append([u, v, w])

        x_list = [x[2] for x in vertices_all]
        self.min_x = min(x_list) # Smallest x value in vertices
        self.max_x = max(x_list) # Largest x value in vertices

        y_list = [y[2] for y in vertices_all]
        self.min_y = min(y_list) # Smallest y value in vertices
        self.max_y = max(y_list) # Largest y value in vertices

        z_list = [z[2] for z in vertices_all]
        self.min_z = min(z_list) # Smallest z value in vertices
        self.max_z = max(z_list) # Largest z value in vertices


        objects = []  # List to hold all the objects

        with open(file_path, 'r') as f:  # Open the .obj file for reading
            current_obj = None  # Will store the current object being processed

            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):  # Skip empty lines and comments
                    continue

                parts = line.split()

                # When encountering a new object (o)
                if parts[0] == 'o':
                    if current_obj:  # If there's an object being processed, add it to the list
                        objects.append(current_obj)

                    # Start a new object
                    current_obj = {
                        'obj_name': parts[1],  # Object name from 'o'
                        'vert_obj_id': [],  # List to store the vertices for the object
                        'textcoords_obj_id': [],  # List to store texture coordinates
                        'normal_obj_id': [],  # List to store the normals for the object
                        'texture_name': None,  # Texture name (if any)
                    }

                # Process texture name (use 'usemtl' to get texture name if available)
                elif parts[0] == 'usemtl':
                    current_obj['texture_name'] = parts[1]

                # Process faces (f)
                elif parts[0] == 'f':  # Face definition
                    # Parse face indices (converting from 1-based to 0-based)
                    vert_id = [int(part.split('/')[0]) - 1 for part in parts[1:]]
                    text_id = [int(part.split('/')[1]) - 1 for part in parts[1:]]
                    normal_id = [int(part.split('/')[2]) - 1 for part in parts[1:]]

                    if len(vert_id) == 3:
                        current_obj['vert_obj_id'].append(vert_id)
                        current_obj['normal_obj_id'].append(normal_id)
                        current_obj['textcoords_obj_id'].append(text_id)
                    elif len(vert_id) == 4:  # Quad face
                        current_obj['vert_obj_id'].append([vert_id[0], vert_id[1], vert_id[2]])  # First triangle
                        current_obj['vert_obj_id'].append([vert_id[0], vert_id[2], vert_id[3]])  # Second triangle

                        current_obj['textcoords_obj_id'].append([text_id[0], text_id[1], text_id[2]])
                        current_obj['textcoords_obj_id'].append([text_id[0], text_id[2], text_id[3]])

                        current_obj['normal_obj_id'].append([normal_id[0], normal_id[1], normal_id[2]])
                        current_obj['normal_obj_id'].append([normal_id[0], normal_id[2], normal_id[3]])

            # Append the last object after finishing the file
            if current_obj:
                objects.append(current_obj) # [[[1,1,1],[1,1,2]], [[2,2,2],[3,3,3]]]

        return vertices_all, texcoords_all, normals_all, objects

    def load_materials(self, file_path):
        materials = {}
        current_material = None

        file_path = file_path[:-3] + 'mtl'

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
                    elif keyword == 'map_Ks':
                        parts = value.split()
                        mat['map_Ks'] = parts[-1]
                    elif keyword == 'map_refl':
                        parts = value.split()
                        mat['map_refl'] = parts[-1]
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
                    normals.append(self.texcoords[int(normal_id)])

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

    def draw(self, model, view, projection):
        for subobj in self.subobjs:
            subobj.draw(model, view, projection)