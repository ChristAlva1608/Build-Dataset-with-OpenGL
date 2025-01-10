import glfw
import numpy as np
from OpenGL.GL import *
import ctypes
from PIL import Image

from libs.shader import *
from libs.transform import *
from libs.buffer import *
from libs.camera import *
import glm
from itertools import cycle
from subObj import *

class Obj:
    def __init__(self, shader, file_path):
        self.shader = shader
        self.vao = VAO()
        self.uma = UManager(self.shader)
        self.subobjs = []
        self.vertices, self.texcoords, self.objects = self.parse_obj_file(file_path)
        self.materials = self.load_materials(file_path)
        self.split_obj()

    def parse_obj_file(self,file_path):
        vertices_all = []
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

                elif parts[0] == 'vt':  # Texture coordinate definition
                    # Parse the texture coordinates (u, v, [w])
                    u = float(parts[1])
                    v = float(parts[2])
                    w = float(parts[3]) if len(parts) > 3 else 0.0  # Default w to 0.0 if not provided
                    texcoords_all.append([u, v, w])

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

                    if len(vert_id) == 3:
                        current_obj['vert_obj_id'].append(vert_id)
                        current_obj['textcoords_obj_id'].append(text_id)
                    elif len(vert_id) == 4:  # Quad face
                        current_obj['vert_obj_id'].append([vert_id[0], vert_id[1], vert_id[2]])  # First triangle
                        current_obj['vert_obj_id'].append([vert_id[0], vert_id[2], vert_id[3]])  # Second triangle

                        current_obj['textcoords_obj_id'].append([text_id[0], text_id[1], text_id[2]])
                        current_obj['textcoords_obj_id'].append([text_id[0], text_id[2], text_id[3]])

            # Append the last object after finishing the file
            if current_obj:
                objects.append(current_obj) # [[[1,1,1],[1,1,2]], [[2,2,2],[3,3,3]]]

        return vertices_all, texcoords_all, objects

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
        img_path = 'patch/textured/image/ledinh.jpeg'
        for obj in self.objects:
            vertices = []
            tecos = []

            for sublist in obj['vert_obj_id']: # [[1 2 3][2 3 4 ]]
                for vert_id in sublist:
                    vertices.append(self.vertices[int(vert_id)])

            for sublist in obj['textcoords_obj_id']:
                for teco_id in sublist:
                    tecos.append(self.texcoords[int(teco_id)])

            if (len(self.subobjs)>=94):
                continue
            model = SubObj( self.shader,
                            vertices,
                            tecos,
                            self.materials[obj['texture_name']],
                            img_path
                            ).setup()
            self.subobjs.append(model)
        print('length of subobjs: ', len(self.subobjs))
        return self

    def setup(self):
        return self 
    
    def update_shader(self, shader):
        for subobj in self.subobjs:
            subobj.update_shader(shader)

    def draw(self, model, view, projection):
        self.vao.activate()
        glUseProgram(self.shader.render_idx)
        
        for subobj in self.subobjs:
            subobj.draw(model, view, projection)

        # Clear the object for new loop
        # print('length of subobjs before clear', len(self.subobjs))
        # self.subobjs.clear()
        # print('length of subobjs after clear', len(self.subobjs))

    # def draw(self, model, view, projection):
    #     self.vao.activate()
    #     glUseProgram(self.shader.render_idx)

    #     glActiveTexture(GL_TEXTURE0)
    #     glBindTexture(GL_TEXTURE_2D, self.texture_id)
    #     glUniform1i(glGetUniformLocation(self.shader.render_idx, "texture_diffuse"), 0)

    #     self.uma.upload_uniform_matrix4fv(np.array(model), 'model', True)
    #     self.uma.upload_uniform_matrix4fv(np.array(view), 'view', True)
    #     self.uma.upload_uniform_matrix4fv(np.array(projection), 'projection', True)

    #     glDrawArrays(GL.GL_TRIANGLES, 0, len(self.vertices)*3)
    #     self.vao.deactivate()


