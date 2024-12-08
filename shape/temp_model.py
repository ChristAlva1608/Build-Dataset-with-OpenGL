from libs.shader import *
from libs.buffer import *
from libs import transform as T
from shape.texture import *
from shape.mesh import *

import glm
import numpy as np
import trimesh
import pyassimp
import os

class Obj:
    def __init__(self, file_path):
        self.vao = VAO()
        self.shader = None
        self.uma = None
        self.meshes = []
        self.materials = {}
        self.textures_loaded = []
        self.vertices, self.normals, self.texcoords, self.indices, self.materials = self.load_obj(file_path)

    def load_obj(self, file_path):
        # Load the OBJ file
        scene = trimesh.load(file_path, force='scene')
        
        # Check if it's a Scene with multiple meshes
        if isinstance(scene, trimesh.Scene):
            for name, mesh in scene.geometry.items():
                print(name)
                self.meshes.append(self.process_mesh(mesh, scene))
                print('hi')
        exit()

    def process_mesh(self, mesh, scene):
        vertices = mesh.vertices
        normals = mesh.vertex_normals
        indices = mesh.faces
        texcoords = None
        if hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None:
            zeros_column = np.zeros((mesh.visual.uv.shape[0], 1))
            texcoords = np.hstack((mesh.visual.uv, zeros_column))
        material = mesh.visual.material
        print(texcoords is not None)
        print(material.image is not None)
        texture = Texture()
        texture.load(material.image)

        return Mesh(vertices, normals, indices, texcoords, texture)

    def update_shader(self, shader):
        self.shader = shader

    def update_uma(self, uma):
        self.uma = uma

    def draw(self):
        for i in range(len(self.meshes)):
            self.meshes[i].draw()