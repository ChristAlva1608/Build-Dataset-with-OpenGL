from libs.shader import *
from libs.buffer import *
from libs import transform as T
import glm
import numpy as np
from mesh3d import *
import pywavefront

class Obj:
    def __init__(self, shader, file_path):
        self.shader = shader
        self.uma = UManager(shader)
        self.materials = {}
        self.meshes = {}
        self.file_path = file_path
        self.load_materials(file_path)
        self.load_obj(file_path)
        self.model = None
        self.view = None
        self.projection = None
    
    def load_obj(self, file_path):
        # Load the .obj file
        scene = pywavefront.Wavefront(file_path, collect_faces=True)

        # Access meshes and their texture coordinates
        for mesh_name, mesh in scene.meshes.items():
            # Get the material for this mesh
            material = mesh.materials[0]  # Assuming one material per mesh
            
            # Get the material name 
            name = material.name

            # Extract vertices (positions)
            data = np.array(material.vertices)  # [vt_x1, vt_y1, vn_x1, vn_y1, vn_z1, v_x1, v_y1, v_z1 ...]
            
            # Number of elements in each group (2 for texcoords, 3 for normals, 3 for vertices)
            num_texcoords = 2
            num_normals = 3
            num_vertices = 3

            # Reshape the data into a 2D array, where each row contains [texcoord_x, texcoord_y, normal_x, normal_y, normal_z, vertex_x, vertex_y, vertex_z]
            data_reshaped = data.reshape(-1, num_texcoords + num_normals + num_vertices)

            # Extract texcoords, normals, and vertices
            texcoords = data_reshaped[:, :num_texcoords]
            normals = data_reshaped[:, num_texcoords:num_texcoords + num_normals]
            vertices = data_reshaped[:, num_texcoords + num_normals:]
            
            # Extract faces (list of indices)
            faces = np.array(mesh.faces)
            
            self.meshes[mesh_name] = Mesh(self.shader, vertices, normals, texcoords, faces, self.materials[name]).setup()
    
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

    # def load_materials(self, file_path):
    #     self.materials = {}
    #     current_material = None
        
    #     # Change from .obj to .mtl
    #     file_path = file_path[:-3] + 'mtl'

    #     def parse_vector(values):
    #         # Convert space-separated string of numbers into list of floats
    #         return [float(x) for x in values.strip().split()]
        
    #     with open(file_path, 'r') as file:
    #         for line in file:
    #             line = line.strip()
    #             if not line or line.startswith('#'):
    #                 continue
                    
    #             parts = line.strip().split(maxsplit=1)
    #             if len(parts) < 2:
    #                 continue
                    
    #             keyword, value = parts
                
    #             if keyword == 'newmtl':
    #                 current_material = value
    #                 self.materials[current_material] = {
    #                     'Ns': None,      # Specular exponent
    #                     'Ni': None,      # Optical density
    #                     'd': None,       # Dissolve
    #                     'Tr': None,      # Transparency
    #                     'Tf': None,      # Transmission filter
    #                     'illum': None,   # Illumination model
    #                     'Ka': None,      # Ambient color
    #                     'Kd': None,      # Diffuse color
    #                     'Ks': None,      # Specular color
    #                     'Ke': None,      # Emissive color
    #                     'map_Kd': None,  # Diffuse texture
    #                     'map_Ks': None,  # Specular texture
    #                     'map_Ka': None,  # Ambient texture
    #                     'map_bump': None # Bump map
    #                 }
                
    #             elif current_material is not None:
    #                 mat = self.materials[current_material]
                    
    #                 if keyword == 'Ns':
    #                     mat['Ns'] = float(value)
    #                 elif keyword == 'Ni':
    #                     mat['Ni'] = float(value)
    #                 elif keyword == 'd':
    #                     mat['d'] = float(value)
    #                 elif keyword == 'Tr':
    #                     mat['Tr'] = float(value)
    #                 elif keyword == 'Tf':
    #                     mat['Tf'] = parse_vector(value)
    #                 elif keyword == 'illum':
    #                     mat['illum'] = int(value)
    #                 elif keyword == 'Kd':
    #                     mat['Kd'] = parse_vector(value)
    #                 elif keyword == 'Ks':
    #                     mat['Ks'] = parse_vector(value)
    #                 elif keyword == 'Ka':
    #                     mat['Ka'] = parse_vector(value)
    #                 elif keyword == 'Ke':
    #                     mat['Ke'] = parse_vector(value)
    #                 elif keyword == 'map_Kd':
    #                     parts = value.split()
    #                     mat['map_Kd'] = parts[-1]
    #                 elif keyword == 'map_Ks':
    #                     parts = value.split()
    #                     mat['map_Ks'] = parts[-1]
    #                 elif keyword == 'map_Ka':
    #                     parts = value.split()
    #                     mat['map_Ka'] = parts[-1]  
    #                 elif keyword == 'map_bump':
    #                     # Handle bump map with potential -bm parameter
    #                     parts = value.split()
    #                     if '-bm' in parts:
    #                         bm_index = parts.index('-bm')
    #                         # Store both the bump multiplier and filename
    #                         mat['map_bump'] = {
    #                             'multiplier': float(parts[bm_index + 1]),
    #                             'filename': parts[-1]
    #                         }
    #                     else:
    #                         mat['map_bump'] = {'filename': parts[-1], 'multiplier': 1.0}

    #     output_file = 'materials.txt'
    #     with open(output_file, 'w') as f:
    #         for material_name, properties in self.materials.items():
    #             f.write(f"Material: {material_name}\n")
    #             for key, value in properties.items():
    #                 f.write(f"  {key}: {value}\n")
    #             f.write("\n")

    def setup(self):
        return self
    
    def update_shader(self, shader):
        for mesh_name, mesh in self.meshes.items():
            mesh.update_shader(shader)

    def draw(self):
        for mesh_name, mesh in self.meshes.items():
            mesh.draw(self.model, self.view, self.projection)