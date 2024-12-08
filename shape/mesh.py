from libs.shader import *
from libs.buffer import *
from libs import transform as T
from shape.texture import *
from shape.mesh import *

import glm
import numpy as np
import trimesh
import os

class Mesh:
    def __init__(self, vertices, normals, indices, texcoords, texture):
        self.vao = VAO()
        self.shader = None
        self.uma = None
        self.vertices = vertices
        self.normals = normals
        self.indices = indices
        self.texcoords = texcoords
        self.texture = texture

    def update_shader(self, shader):
        self.shader = shader

    def update_uma(self, uma):
        self.uma = uma

    def setup(self):
        self.vao.add_vbo(0, self.vertices, ncomponents=3, stride=0, offset=None)
        if self.normals is not None:
            self.vao.add_vbo(2, self.normals, ncomponents=3, stride=0, offset=None)
        if self.texcoords is not None:
            self.vao.add_vbo(1, self.texcoords, ncomponents=2, stride=0, offset=None)

        self.vao.add_ebo(self.indices)

        GL.glUseProgram(self.shader.render_idx)

        self.model = glm.mat4(1.0)
        
        camera_pos = glm.vec3(0.0, 0.0, 0.0)
        camera_target = glm.vec3(0.0, 0.0, 0.0)
        up_vector = glm.vec3(0.0, 1.0, 0.0)
        
        self.view = glm.lookAt(camera_pos, camera_target, up_vector)
        self.projection = glm.perspective(glm.radians(45.0), 1200.0 / 800.0, 0.1, 100.0)

        model_view_matrix = self.view * self.model 
        
        self.uma.upload_uniform_matrix4fv(np.array(model_view_matrix), 'modelview', True)
        self.uma.upload_uniform_matrix4fv(np.array(self.projection), 'projection', True)

        I_light = np.array([
            [0.9, 0.4, 0.6],
            [0.9, 0.4, 0.6],
            [0.9, 0.4, 0.6],
        ], dtype=np.float32)
        light_pos = np.array([0, 0.5, 0.9], dtype=np.float32)

        self.uma.upload_uniform_matrix3fv(I_light, 'I_light', False)
        self.uma.upload_uniform_vector3fv(light_pos, 'light_pos')

        if self.materials:
            first_material = next(iter(self.materials.values()))
            K_materials = np.array([
                first_material['Kd'],
                first_material['Ks'],
                first_material['Ka'],
            ], dtype=np.float32)
            
            shininess = first_material['Ns']
            
            # Upload texture flag if material has a texture
            has_texture = 1 if first_material.get('map_Kd') else 0
            self.uma.upload_uniform_scalar1i(has_texture, 'has_texture')
        else:
            K_materials = np.array([
                [0.6, 0.4, 0.7],
                [0.6, 0.4, 0.7],
                [0.6, 0.4, 0.7],
            ], dtype=np.float32)
            shininess = 100.0
            self.uma.upload_uniform_scalar1i(0, 'has_texture')

        self.uma.upload_uniform_matrix3fv(K_materials, 'K_materials', False)
        self.uma.upload_uniform_scalar1f(shininess, 'shininess')
        self.uma.upload_uniform_scalar1i(1, 'mode')

        return self

    def draw(self, shader):
        GL.glActiveTexture(GL.GL_TEXTURE0)  
        self.uma.upload_uniform_scalar1i(0,'texture_diffuse'); 
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture)
    
        # draw mesh
        self.vao.activate()
        GL.glDrawElements(GL.GL_TRIANGLES, len(self.indices), GL.GL_UNSIGNED_INT, None)
        
        GL.glActiveTexture(GL.GL_TEXTURE0)