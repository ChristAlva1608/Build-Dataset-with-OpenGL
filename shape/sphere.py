from libs.shader import *
from libs.buffer import *
from PIL import Image
import glm
import numpy as np
import math

class Sphere:
    def __init__(self, shader, radius=0.1, sectors=20, stacks=20):
        self.radius = radius
        self.sectors = sectors
        self.stacks = stacks
        self.shader = shader
        self.uma = UManager(self.shader)
        self.vao = VAO()

        self.generate_sphere()

    def generate_sphere(self):

        vertices = []
        normals = []
        indices = []

        sector_step = 2 * math.pi / self.sectors
        stack_step = math.pi / self.stacks

        for i in range(self.stacks + 1):
            stack_angle = math.pi / 2 - i * stack_step
            xz = self.radius * math.cos(stack_angle)
            y = self.radius * math.sin(stack_angle)

            for j in range(self.sectors + 1):
                sector_angle = j * sector_step

                x = xz * math.cos(sector_angle)
                z = xz * math.sin(sector_angle)

                vertices.append([x, y, z])
                normals.append([x/self.radius, y/self.radius, z/self.radius])

        for i in range(self.stacks):
            k1 = i * (self.sectors + 1)
            k2 = k1 + self.sectors + 1

            for j in range(self.sectors):
                if i != 0:
                    indices.extend([k1, k2, k1 + 1])
                if i != (self.stacks - 1):
                    indices.extend([k1 + 1, k2, k2 + 1])

                k1 += 1
                k2 += 1

        self.vertices = np.array(vertices, dtype=np.float32)
        self.normals = np.array(normals, dtype=np.float32)
        self.indices = np.array(indices, dtype=np.uint32)

        # Generate random colors for each vertex
        self.colors = np.tile([1.0, 1.0, 0.0], (len(vertices), 1)).astype(np.float32) # yellow

    def update_model_matrix(self, model):

        # transform all matrix to glm mat4 to use * for matrix multiplication
        if not isinstance(model, glm.mat4):
            model = glm.mat4(*model.flatten())
        if not isinstance(self.model, glm.mat4):
            self.model = glm.mat4(*self.model.flatten())

        self.model = model * self.model
        
    def setup(self):
        self.vao.add_vbo(0, self.vertices, ncomponents=3, stride=0, offset=None)
        self.vao.add_vbo(1, self.colors, ncomponents=3, stride=0, offset=None)
        self.vao.add_vbo(2, self.normals, ncomponents=3, stride=0, offset=None)
        self.vao.add_ebo(self.indices)

        GL.glUseProgram(self.shader.render_idx)

        self.model = glm.mat4(1.0)
        self.view = glm.lookAt(glm.vec3(0, 0, 10), glm.vec3(0, 0, 0), glm.vec3(0, 1, 0))
        self.projection = glm.perspective(glm.radians(45.0), 800.0 / 600.0, 0.1, 100.0)

        model_view_matrix = self.model * self.view

        self.uma.upload_uniform_matrix4fv(np.array(model_view_matrix), 'modelview', True)
        self.uma.upload_uniform_matrix4fv(np.array(self.projection), 'projection', True)

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

    def draw(self):
        GL.glUseProgram(self.shader.render_idx)

        model_view_matrix = self.view * self.model 
        self.uma.upload_uniform_matrix4fv(np.array(model_view_matrix, dtype=np.float32), 'modelview', True)
        self.uma.upload_uniform_matrix4fv(np.array(self.projection, dtype=np.float32), 'projection', True)

        self.vao.activate()
        GL.glDrawElements(GL.GL_TRIANGLE_STRIP, len(self.indices), GL.GL_UNSIGNED_INT, None)