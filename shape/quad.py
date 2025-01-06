from libs.shader import *
from libs.buffer import *
import glm
import numpy as np

class Quad:
    def __init__(self, shader):
        # Quad vertex data (positions and texture coordinates)
        self.vertices = np.array([
            -1.0,  1.0, 0.0,  0.0, 1.0,  # Top-left
            -1.0, -1.0, 0.0,  0.0, 0.0,  # Bottom-left
            1.0, -1.0, 0.0,  1.0, 0.0,  # Bottom-right
            1.0,  1.0, 0.0,  1.0, 1.0   # Top-right
        ], dtype=np.float32)

        # Quad indices for two triangles
        self.indices = np.array([
            0, 1, 2, 
            2, 3, 0
        ], dtype=np.uint32)

        # Initialize VAO, Shader, and Texture ID placeholders
        self.shader = shader
        self.uma = UManager(self.shader)
        self.vao = VAO()
        self.texture_id = None  # Placeholder for the depth or texture ID

    def setup(self):
        """
        Setup the Quad by binding vertices, indices, and shaders.
        Optionally bind a depth texture.
        """
        # Add VBO for vertex positions and texture coordinates
        self.vao.add_vbo(0, self.vertices, ncomponents=2, stride=4 * self.vertices.itemsize, offset=0)
        self.vao.add_vbo(1, self.vertices, ncomponents=2, stride=4 * self.vertices.itemsize, offset=2 * self.vertices.itemsize)

        # Add EBO for indices
        self.vao.add_ebo(self.indices)

        GL.glUseProgram(self.shader.render_idx)

        # Projection and ModelView matrix defaults (Identity)
        self.model = glm.mat4(1.0)
        self.view = glm.mat4(1.0)
        self.projection = glm.mat4(1.0)

        return self

    def draw(self):
        GL.glUseProgram(self.shader.render_idx)

        # Bind the texture
        if self.texture_id:
            GL.glActiveTexture(GL.GL_TEXTURE0)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture_id)

        # Upload matrices to shader
        modelview = self.view * self.model 
        self.uma.upload_uniform_matrix4fv(np.array(modelview, dtype=np.float32), 'modelview', True)
        self.uma.upload_uniform_matrix4fv(np.array(self.projection, dtype=np.float32), 'projection', True)

        # Activate VAO and render
        self.vao.activate()
        GL.glDrawElements(GL.GL_TRIANGLES, len(self.indices), GL.GL_UNSIGNED_INT, None)