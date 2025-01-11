from libs.shader import *
from libs.buffer import *
from libs import transform as T
from shape.texture import *
from enum import Enum

import glm
import numpy as np
from OpenGL.GL import *

class TextureMap(Enum):
    KA = ("map_Ka", "texture_ambient")
    KD = ("map_Kd", "texture_diffuse")
    KS = ("map_Ks", "texture_specular")
    REFL = ("map_refl", "texture_refl")
    BUMP = ("map_bump", "texture_bump")

class SubObj:
    def __init__(self, shader, vert, teco , material, img_path):
        self.shader = shader
        self.vao = VAO()
        self.uma = UManager(self.shader)

        self.vertices = np.array(vert, dtype=np.float32)
        self.textcoords = np.array(teco, dtype=np.float32)

        self.materials = material
        self.map_Kd_flag = False
        self.map_Ka_flag = False
        self.map_Ks_flag = False
        self.map_refl_flag = False
        self.map_bump_flag = False
        self.texture_id = {}
        self.texture_flags = {}

        # Iterate through all possible texture maps defined in the enum
        for texture in TextureMap:
            map_key, uniform_name = texture.value
            self.texture_flags[map_key] = False  # Initialize texture flag as False
            if map_key in material and material[map_key]:
                if map_key == 'map_bump':
                    texture_path = material[map_key]['filename']
                else:
                    texture_path = material[map_key]

                if os.path.exists(texture_path):
                    self.texture_flags[map_key] = True
                    self.texture_id[map_key] = self.uma.setup_texture(uniform_name, texture_path)

    def update_shader(self, shader):
        self.shader = shader
        self.uma = UManager(self.shader)

    def setup(self):
        self.vao.add_vbo(0, self.vertices, ncomponents=3, dtype=GL.GL_FLOAT, normalized=False, stride=0, offset=None)
        self.vao.add_vbo(2, self.textcoords, ncomponents=3, dtype=GL.GL_FLOAT, normalized=False, stride=0, offset=None)

        # Light setup (you can modify these values)
        I_light = np.array([
            [0.9, 0.4, 0.6],  # diffuse
            [0.9, 0.4, 0.6],  # specular
            [0.9, 0.4, 0.6],  # ambient
        ], dtype=np.float32)
        light_pos = np.array([0, 0.5, 0.9], dtype=np.float32)

        self.uma.upload_uniform_matrix3fv(I_light, 'I_light', False)
        self.uma.upload_uniform_vector3fv(light_pos, 'light_pos')

        # Materials setup (you can modify these values)
        self.K_materials = np.array([
            self.materials['Kd'],
            self.materials['Ks'],
            self.materials['Ka'],
        ], dtype=np.float32)

        self.shininess = self.materials['Ns']

        # self.K_materials = np.array([
        #     [0.6, 0.4, 0.7],  # diffuse
        #     [0.6, 0.4, 0.7],  # specular
        #     [0.6, 0.4, 0.7],  # ambient
        # ], dtype=np.float32)
        
        # self.shininess = 100.0

        self.uma.upload_uniform_matrix3fv(self.K_materials, 'K_materials', False)

        mode = 1

        self.uma.upload_uniform_scalar1f(self.shininess, 'shininess')
        self.uma.upload_uniform_scalar1i(mode, 'mode')
        return self

    def draw(self, model, view, projection):
        self.vao.activate()
        glUseProgram(self.shader.render_idx)

        # Bind and upload textures dynamically
        for idx, texture in enumerate(TextureMap):
            map_key, uniform_name = texture.value
            if self.texture_flags[map_key]:
                glActiveTexture(GL_TEXTURE0 + idx)
                glBindTexture(GL_TEXTURE_2D, self.texture_id[map_key])
                glUniform1i(glGetUniformLocation(self.shader.render_idx, uniform_name), idx)

        self.uma.upload_uniform_matrix4fv(np.array(model), 'model', True)
        self.uma.upload_uniform_matrix4fv(np.array(view), 'view', True)
        self.uma.upload_uniform_matrix4fv(np.array(projection), 'projection', True)

        glDrawArrays(GL.GL_TRIANGLES, 0, len(self.vertices)*3)

        # Unbind all textures
        for idx, texture in enumerate(TextureMap):
            glActiveTexture(GL_TEXTURE0 + idx)
            glBindTexture(GL_TEXTURE_2D, 0)

        self.vao.deactivate()