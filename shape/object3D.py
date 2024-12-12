import pywavefront.material
import pywavefront.wavefront
from libs.shader import *
from libs.buffer import *
from libs import transform as T
import glm
import numpy as np
import trimesh
import pywavefront
from mesh import *

class Obj:
    def __init__(self, shader, file_path):
        self.vao = VAO()
        self.shader = shader
        self.uma = UManager(self.shader)
        self.materials = {}
        self.meshes = {}
        self.load_materials(file_path)
        self.parse_obj_file(file_path)

    def parse_obj_file(self, file_path):
        current_mesh = None
        vertices = []
        normals = []
        texcoords = []
        faces = []
        material = None

        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip()
                parts = line.split()

                if not line or (line.startswith('#') and len(parts)==1):  # Ignore comments and empty lines
                    continue

                if line.startswith('#') and len(parts)>1:
                    if parts[1] == 'object' and current_mesh:  # Store the previous mesh
                        self.meshes[current_mesh] = Mesh(self.shader, np.array(vertices), np.array(normals), np.array(texcoords), np.array(faces), self.materials[material]).setup()
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
                    texcoords.append(list(map(float, parts[1:])))


                elif prefix == 'usemtl':  # Material
                    material = parts[1]

                elif prefix == 'f':  # Face
                    face = []
                    for vertex in parts[1:]:
                        # Take only the vertex index (before first slash)
                        index = int(vertex.split('/')[0])
                        face.append(index)


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
        if current_mesh:
            self.meshes[current_mesh] = Mesh(self.shader, np.array(vertices), np.array(normals), np.array(texcoords), np.array(faces), self.materials[material]).setup()

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

    def update_shader(self, shader):
        self.shader = shader
        self.uma = UManager(self.shader)

    def draw(self):
        for mesh_name, mesh in self.meshes.items():
            mesh.draw()