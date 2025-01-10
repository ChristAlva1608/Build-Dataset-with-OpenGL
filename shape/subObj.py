from libs.shader import *
from libs.buffer import *
from libs import transform as T
from shape.texture import *

import glm
import numpy as np
from OpenGL.GL import *

class SubObj:
    def __init__(self, shader, vert, teco , material, img_path):
        self.shader = shader
        self.vao = VAO()
        self.uma = UManager(self.shader)

        self.vertices = np.array(vert, dtype=np.float32)
        self.textcoords = np.array(teco, dtype=np.float32)
        image_path = img_path

        if material['map_Ka'] is not None:
            image_path = material['map_Ka']

        image = Image.open(image_path)
        image = image.transpose(Image.FLIP_TOP_BOTTOM)  # Flip image for OpenGL
        img_data = np.array(image, dtype=np.uint8)

        self.texture_id = self.uma.setup_texture('texture_diffuse', img_data)
        # # Generate texture ID and bind it
        # self.texture_id = glGenTextures(1)
        # glBindTexture(GL_TEXTURE_2D, self.texture_id)
        # # Set texture parameters
        # glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        # glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        # glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        # glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        # # Load the texture into OpenGL
        # glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image.width, image.height, 0, GL_RGB, GL_UNSIGNED_BYTE, img_data)

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
        K_materials = np.array([
            [0.6, 0.4, 0.7],  # diffuse
            [0.6, 0.4, 0.7],  # specular
            [0.6, 0.4, 0.7],  # ambient
        ], dtype=np.float32)

        self.uma.upload_uniform_matrix3fv(K_materials, 'K_materials', False)

        shininess = 100.0
        mode = 1

        self.uma.upload_uniform_scalar1f(shininess, 'shininess')
        self.uma.upload_uniform_scalar1i(mode, 'mode')

        return self

    def draw(self, model, view, projection):
        self.vao.activate()
        glUseProgram(self.shader.render_idx)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        glUniform1i(glGetUniformLocation(self.shader.render_idx, "texture_diffuse"), 0)

        self.uma.upload_uniform_matrix4fv(np.array(model), 'model', True)
        self.uma.upload_uniform_matrix4fv(np.array(view), 'view', True)
        self.uma.upload_uniform_matrix4fv(np.array(projection), 'projection', True)

        glDrawArrays(GL.GL_TRIANGLES, 0, len(self.vertices)*3)
        self.vao.deactivate()