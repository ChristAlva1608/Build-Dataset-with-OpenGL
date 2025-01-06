from libs.shader import *
from libs.buffer import *
from libs import transform as T
from shape.texture import *
from shape.mesh import *
import glm
import numpy as np
import trimesh
import os

class Scene:
    def __init__(self, shader, scene_path, mtl_path):
        self.vao = VAO()
        self.shader = shader
        self.uma = UManager(self.shader)
        self.meshes = []
        self.materials = {}
        self.textures_loaded = []
        self.load_obj(scene_path)
        # self.vertices, self.normals, self.texcoords, self.indices, self.materials = self.load_obj(scene_path)

    def load_obj(self, scene_path):
        # Load the OBJ file
        scene = trimesh.load(scene_path, force='scene')
        
        # Check if it's a Scene with multiple meshes
        if isinstance(scene, trimesh.Scene):
            for name, mesh in scene.geometry.items():
                print("name: ", name)
                self.meshes.append(self.process_mesh(mesh))
        print("length of meshes: ", len(self.meshes))
        # exit()

    def process_mesh(self, mesh):
        vertices = mesh.vertices
        normals = mesh.vertex_normals
        indices = mesh.faces
        texcoords = None
        if hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None:
            zeros_column = np.zeros((mesh.visual.uv.shape[0], 1))
            texcoords = np.hstack((mesh.visual.uv, zeros_column))
        material = mesh.visual.material
        print("texcoords is not None: ", texcoords is not None)
        print("material.image is not None" ,material.image is not None)
        texture = Texture()
        texture.load(material.image)

        print("type of shader", type(self.shader))
        return Mesh(self.shader, vertices, normals, indices, texcoords, texture).setup()

    def update_shader(self, shader):
        self.shader = shader
        self.uma = UManager(self.shader)
        
        for mesh in self.meshes:
            mesh.update_shader(self.shader)
            mesh.setup()

    def draw(self):
        for i in range(len(self.meshes)):
            self.meshes[i].draw()