from libs.shader import *
from libs.buffer import *
from libs import transform as T
import glm
import numpy as np
from mesh import *

class Obj:
    def __init__(self, shader, file_path):
        self.shader = shader
        self.uma = UManager(shader)
        self.materials = {}
        self.meshes = {}
        self.file_path = file_path
        self.load_materials(file_path)
        self.parse_obj_file(file_path)
        self.model = None
        self.view = None
        self.projection = None
    
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
            v_idx, vt_idx, vn_idx = face
                
            # Create a unique key for this vertex combination
            key = (v_idx, vt_idx, vn_idx)
            
            if key not in vertex_dict:
                vertex_dict[key] = current_index
                final_vertices.append(vertices[v_idx])
                final_texcoords.append(texcoords[vt_idx])
                final_normals.append(normals[vn_idx])
                current_index += 1
            
            final_indices.append(vertex_dict[key])
        
        return (final_vertices, final_normals, final_texcoords, final_indices)
    
    def parse_obj_file(self, file_path):
        current_mesh = None
        vertices = []
        normals = []
        texcoords = []
        faces = []
        material = None
        count = 0
        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip()
                parts = line.split()

                if not line or (line.startswith('#') and len(parts)==1):  # Ignore comments and empty lines
                    continue

                if line.startswith('#') and len(parts)>1:
                    if parts[1] == 'object' and current_mesh:  # Store the previous mesh
                        
                        count += 1
                        if count != 1:
                            break

                        # print('vertices:', vertices)
                        # print('normals:', normals)
                        # print('texcoords:', texcoords)
                        # print('faces:', faces)
                        # print('material:', self.materials[material])

                        # faces = list(map(lambda x: x - 1, faces)) # substract for starting index 0
                        vertices, normals, texcoords, indices = self.process_mesh_data(vertices, normals, texcoords, faces) # create a single index for each vertex
                        self.meshes[current_mesh] = Mesh(self.shader, np.array(vertices), np.array(normals), np.array(texcoords), np.array(indices), self.materials[material]).setup()
                        vertices = []
                        normals = []
                        texcoords = []
                        faces = []
                        material = None
                    else: continue

                prefix = parts[0]

                if prefix == 'o':  # New object/group
                    current_mesh = parts[1]

                elif prefix == 'v':  # Vertex
                    vertices.append(list(map(float, parts[1:])))

                elif prefix == 'vn':  # Vertex normal
                    normals.append(list(map(float, parts[1:])))

                elif prefix == 'vt':  # Texture coordinate
                    if (len(parts) == 3):
                        texcoords.append(list(map(float, parts[1:])))
                    elif (len(parts) == 4):
                        texcoords.append(list(map(float, parts[1:-1])))

                elif prefix == 'usemtl':  # Material
                    material = parts[1]

                elif prefix == 'f':  # Face
                    face = []
                    for vertex in parts[1:]:
                        # Take only the vertex index (before first slash)
                        face.append(list(map(lambda x: int(x)-1, vertex.split('/'))))  # [[1,1,1],[2,2,2],[3,3,3]]

                    # If it's a triangle, add directly
                    if len(face) == 3:
                        faces.extend(face)

                    # If it's a quad, triangulate it into two triangles
                    elif len(face) == 4:
                        # First triangle: vertices 0,1,2 
                        faces.extend([face[0], face[1], face[2]])
                        # Second triangle: vertices 0,2,3
                        faces.extend([face[0], face[2], face[3]])

        # Store the last mesh
        # if current_mesh:
        #     print(f'Current mesh: {current_mesh}')
        #     self.meshes[current_mesh] = Mesh(self.shader, np.array(vertices), np.array(normals), np.array(texcoords), np.array(faces), self.materials[material]).setup()

        with open('obj.txt', 'w') as file:
            for mesh_name, mesh_data in self.meshes.items():
                file.write(f"# object {mesh_name}\n")

                # Save vertices
                file.write("# vertices\n")
                for vertex in mesh_data.vertices:
                    file.write(f"v {' '.join(map(str, vertex))}\n")

                # Save normals
                file.write("# vertex normals\n")
                for normal in mesh_data.normals:
                    file.write(f"vn {' '.join(map(str, normal))}\n")

                # Save texture coordinates
                file.write("# texture coordinates\n")
                for texcoord in mesh_data.texcoords:
                    file.write(f"vt {' '.join(map(str, texcoord))}\n")

                for i in range(0, len(mesh_data.indices), 3):
                    # Get 3 indices for this triangle
                    v1 = mesh_data.indices[i]
                    v2 = mesh_data.indices[i + 1]
                    v3 = mesh_data.indices[i + 2]
                    # Add 1 to convert from 0-based to 1-based indexing for OBJ format
                    file.write(f"f {v1+1} {v2+1} {v3+1}\n")

                # Save material
                file.write(f"usemtl {mesh_data.material}\n")
                file.write("\n")

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

    # def load_materials(self, file_path):
    #     obj_file = pywavefront.Wavefront(file_path, create_materials=True)

    #     # Access materials
    #     self.materials = {}
    #     for name, material in obj_file.materials.items():
    #         def trim_list(lst):
    #             return lst[:-1] if isinstance(lst, list) and len(lst) == 4 else lst

    #         self.materials[name] = {
    #             'Ns': getattr(material, 'shininess', None),                                 # Specular exponent
    #             'Ni': getattr(material, 'optical_density', None),                           # Optical density
    #             'd': getattr(material, 'transparency', None),                               # Dissolve
    #             'Tr': 1 - getattr(material, 'transparency', 1.0),                           # Transparency (inverse dissolve)
    #             'Tf': getattr(material, 'transmission_filter', [1, 1, 1]),                  # Transmission filter
    #             'illum': getattr(material, 'illumination_model', None),                     # Illumination model
    #             'Ka': trim_list(getattr(material, 'ambient', [0, 0, 0])),                   # Ambient color
    #             'Kd': trim_list(getattr(material, 'diffuse', [0, 0, 0])),                   # Diffuse color
    #             'Ks': trim_list(getattr(material, 'specular', [0, 0, 0])),                  # Specular color
    #             'Ke': trim_list(getattr(material, 'emissive', [0, 0, 0])),                  # Emissive color
    #             'map_Ka': getattr(material, 'texture_ambient', None),                       # Ambient texture
    #             'map_Kd': getattr(material, 'texture', None),                               # Diffuse texture
    #             'map_bump': getattr(material, 'texture_bump', None)                         # Bump map
    #         }

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