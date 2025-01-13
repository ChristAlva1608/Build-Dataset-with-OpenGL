from libs.shader import *
from libs.buffer import *
from libs import transform as T
import glm
import numpy as np
import trimesh
import os
import pywavefront

class Object:
    def __init__(self, shader, file_path):
        self.vao = VAO()
        self.shader = shader
        self.uma = UManager(self.shader)
        # self.materials = self.load_materials(file_path)
        self.vertices, self.normals, self.texcoords, self.indices = self.load_obj(file_path)

    def process_mesh_data(self, vertices, normals, texcoords, faces):
        """Process mesh data to create indexed arrays"""
        final_vertices = []
        final_normals = []
        final_texcoords = []
        final_indices = []
        
        # Create a vertex data dictionary to handle unique combinations
        vertex_dict = {}
        current_index = 0
        
        for face in faces:
            face_vertices, face_texcoords, face_normals = face
            
            # Process each vertex in the face
            for i in range(len(face_vertices)):
                v_idx = face_vertices[i]
                vt_idx = face_texcoords[i] if face_texcoords else -1
                vn_idx = face_normals[i] if face_normals else -1
                
                # Create a unique key for this vertex combination
                key = (v_idx, vt_idx, vn_idx)
                
                if key not in vertex_dict:
                    vertex_dict[key] = current_index
                    final_vertices.append(vertices[v_idx])
                    
                    if vt_idx != -1 and texcoords.size > 0:
                        final_texcoords.append(texcoords[vt_idx])
                    elif texcoords.size > 0:
                        final_texcoords.append([0.0, 0.0])
                        
                    if vn_idx != -1 and normals.size > 0:
                        final_normals.append(normals[vn_idx])
                    
                    current_index += 1
                
                final_indices.append(vertex_dict[key])
        
        return (np.array(final_vertices, dtype=np.float32),
                np.array(final_normals, dtype=np.float32) if final_normals else None,
                np.array(final_texcoords, dtype=np.float32) if final_texcoords else None,
                np.array(final_indices, dtype=np.uint32))
    
    def load_materials(self, file_path):
        self.materials = {}
        current_material = None
        
        # Change from .obj to .mtl
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
                    self.materials[current_material] = {
                        'Ns': None,      # Specular exponent
                        'Ni': None,      # Optical density
                        'd': None,       # Dissolve
                        'Tr': None,      # Transparency
                        'Tf': None,      # Transmission filter
                        'illum': None,   # Illumination model
                        'Ka': None,      # Ambient color
                        'Kd': None,      # Diffuse color
                        'Ks': None,      # Specular color
                        'Ke': None,      # Emissive color
                        'map_Kd': None,  # Diffuse texture
                        'map_Ks': None,  # Specular texture
                        'map_Ka': None,  # Ambient texture
                        'map_bump': None # Bump map
                    }
                
                elif current_material is not None:
                    mat = self.materials[current_material]
                    
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
                    elif keyword == 'Kd':
                        mat['Kd'] = parse_vector(value)
                    elif keyword == 'Ks':
                        mat['Ks'] = parse_vector(value)
                    elif keyword == 'Ka':
                        mat['Ka'] = parse_vector(value)
                    elif keyword == 'Ke':
                        mat['Ke'] = parse_vector(value)
                    elif keyword == 'map_Kd':
                        parts = value.split()
                        mat['map_Kd'] = parts[-1]
                    elif keyword == 'map_Ks':
                        parts = value.split()
                        mat['map_Ks'] = parts[-1]
                    elif keyword == 'map_Ka':
                        parts = value.split()
                        mat['map_Ka'] = parts[-1]  
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

        output_file = 'materials.txt'
        with open(output_file, 'w') as f:
            for material_name, properties in self.materials.items():
                f.write(f"Material: {material_name}\n")
                for key, value in properties.items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")

    def parse_obj_file(self, file_path):
        vertices = []
        normals = []
        texcoords = []
        faces = []
        
        with open(file_path, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                    
                values = line.split()
                if not values:
                    continue
                    
                if values[0] == 'v':
                    vertices.append([float(x) for x in values[1:4]])
                elif values[0] == 'vt':
                    texcoords.append([float(x) for x in values[1:3]])
                elif values[0] == 'vn':
                    normals.append([float(x) for x in values[1:4]])
                elif values[0] == 'f':
                    # Handle different face formats
                    face_vertices = []
                    face_texcoords = []
                    face_normals = []
                    
                    for v in values[1:]:
                        w = v.split('/')
                        face_vertices.append(int(w[0]) - 1)
                        if len(w) > 1 and w[1]:  # Check if texture coordinate index exists
                            face_texcoords.append(int(w[1]) - 1)
                        if len(w) > 2 and w[2]:  # Check if normal index exists
                            face_normals.append(int(w[2]) - 1)
                            
                    faces.append((face_vertices, face_texcoords, face_normals))
        
        return np.array(vertices), np.array(normals), np.array(texcoords), faces

    def load_obj(self, file_path):
        try:
            # First try to load with trimesh
            scene = trimesh.load(file_path, process=False)
            if isinstance(scene, trimesh.Scene):
                scene = scene.to_mesh()
            
            vertices = scene.vertices.astype(np.float32)
            normals = scene.vertex_normals.astype(np.float32)
            indices = scene.faces.astype(np.uint32)

            # Try to get texture coordinates from visual
            texcoords = None
            if hasattr(scene.visual, 'uv') and scene.visual.uv is not None:
                texcoords = scene.visual.uv.astype(np.float32)
            
            # If no texture coordinates found in visual, try parsing the OBJ file directly
            if texcoords is None:
                raw_vertices, raw_normals, raw_texcoords, faces = self.parse_obj_file(file_path)
                if len(raw_texcoords) > 0:  # If we found texture coordinates
                    vertices, normals, texcoords, indices = self.process_mesh_data(
                        raw_vertices, raw_normals, raw_texcoords, faces)
            
            return vertices, normals, texcoords, indices
            
        except Exception as e:
            print(f"Error loading OBJ file: {e}")
            # Fallback to direct OBJ parsing if trimesh fails
            raw_vertices, raw_normals, raw_texcoords, faces = self.parse_obj_file(file_path)
            vertices, normals, texcoords, indices = self.process_mesh_data(
                raw_vertices, raw_normals, raw_texcoords, faces)
            
            return vertices, normals, texcoords, indices

    def update_shader(self, shader):
        self.shader = shader
        self.uma = UManager(self.shader)

    def setup(self):
        # print('vertices: ', self.vertices[0])
        # print('normals: ', self.normals[0])
        # print('texcoords: ', self.texcoords[0])
        # exit()
        self.vao.add_vbo(0, self.vertices, ncomponents=3, stride=0, offset=None)
        if self.normals is not None:
            self.vao.add_vbo(2, self.normals, ncomponents=3, stride=0, offset=None)
        # if self.texcoords is not None:
        #     self.vao.add_vbo(2, self.texcoords, ncomponents=2, stride=0, offset=None)
        self.vao.add_ebo(self.indices)

        GL.glUseProgram(self.shader.render_idx)
        
        self.model = glm.mat4(1.0)

        camera_pos = glm.vec3(0.0, 0.0, 0.0)
        camera_target = glm.vec3(0.0, 0.0, 0.0)
        up_vector = glm.vec3(0.0, 1.0, 0.0)
        
        # glm.lookAt(glm.vec3(0, 0, 50), glm.vec3(0, 0, 0), glm.vec3(0, 1, 0))
        self.view = glm.lookAt(camera_pos, camera_target, up_vector)
        self.projection = glm.perspective(glm.radians(45.0), 1400.0 / 700.0, 0.1, 100.0)

        I_light = np.array([
            [0.9, 0.4, 0.6],
            [0.9, 0.4, 0.6],
            [0.9, 0.4, 0.6],
        ], dtype=np.float32)
        light_pos = np.array([0, 0.5, 0.9], dtype=np.float32)

        self.uma.upload_uniform_matrix3fv(I_light, 'I_light', False)
        self.uma.upload_uniform_vector3fv(light_pos, 'light_pos')

        self.K_materials = np.array([
            [0.6, 0.4, 0.7],
            [0.6, 0.4, 0.7],
            [0.6, 0.4, 0.7],
        ], dtype=np.float32)
        self.shininess = 100.0

        self.uma.upload_uniform_matrix3fv(self.K_materials, 'K_materials', False)
        self.uma.upload_uniform_scalar1f(self.shininess, 'shininess')
        self.uma.upload_uniform_scalar1i(1, 'mode')

        return self

    def update_Kmat(self, diffuse, specular, ambient):
        self.K_materials = np.array([
            diffuse,
            specular,
            ambient,
        ], dtype=np.float32)

    def update_colormap(self, selected_colormap):
        self.uma.upload_uniform_scalar1i(selected_colormap, 'colormap_selection')

    def update_near_far(self, near, far):
        self.uma.upload_uniform_scalar1f(near, 'near')
        self.uma.upload_uniform_scalar1f(far, 'far')

    def draw(self, model, view, projection):
        
        self.uma.upload_uniform_matrix3fv(self.K_materials, 'K_materials', False)
        self.uma.upload_uniform_scalar1f(self.shininess, 'shininess')

        self.uma.upload_uniform_matrix4fv(np.array(model, dtype=np.float32), 'model', True)
        self.uma.upload_uniform_matrix4fv(np.array(view, dtype=np.float32), 'view', True)
        self.uma.upload_uniform_matrix4fv(np.array(projection, dtype=np.float32), 'projection', True)

        self.vao.activate()
        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
        GL.glDrawElements(GL.GL_TRIANGLES, len(self.indices)*3, GL.GL_UNSIGNED_INT, None)