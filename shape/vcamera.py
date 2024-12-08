from libs.shader import *
from libs.buffer import *
import glm
import numpy as np
import glfw
import ctypes

class VCamera:
    def __init__(self, shader, base_width=0.2, base_height=0.2, height=0.3):
        # Define vertices for the pyramid: 4 base vertices and 1 apex vertex
        self.vertices = np.array([
            # Base vertices (clockwise order)
            [-base_width / 2, -base_height / 2, 0.0],  # Bottom-left
            [base_width / 2, -base_height / 2, 0.0],   # Bottom-right
            [base_width / 2, base_height / 2, 0.0],    # Top-right
            [-base_width / 2, base_height / 2, 0.0],   # Top-left
            # Apex vertex
            [0.0, 0.0, height],                       # Apex
        ], dtype=np.float32)

        self.tex_coords = np.array([
            # Base texture coordinates (assumes a simple square mapping)
            [0.0, 0.0],  # Bottom-left
            [1.0, 0.0],  # Bottom-right
            [1.0, 1.0],  # Top-right
            [0.0, 1.0]   # Top-left
        ], dtype=np.float32)
        
        # Indices to define the faces of the pyramid
        self.indices = np.array([
            # Base (using a triangle fan for simplicity)
            0, 1, 2, 3, 0,
            # Sides (connecting base vertices to the apex)
            0, 4, 1,
            1, 4, 2,
            2, 4, 3,
            3, 4, 0,
        ], dtype=np.uint32)

        # Normals (approximate for visualization, one normal per face)
        self.normals = np.array([
            [0.0, 0.0, -1.0],  # Base normal (downward)
            # Side normals (computed manually or with cross products for better accuracy)
            [-1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [0.0, -1.0, 1.0],
        ], dtype=np.float32)

        # Colors (uniform for simplicity)
        self.colors = np.full(self.vertices.shape, 0.8).astype(np.float32)

        # Initialize shader, VAO, and other parameters
        self.vao = VAO()
        self.shader = shader
        self.uma = UManager(self.shader)

    def load_texture(self, texture):
        self.uma.setup_texture('texture1', texture)

    def setup(self):
        # Setup VAO with VBOs for vertices, normals, and colors if needed
        self.vao.add_vbo(0, self.vertices, ncomponents=3, stride=0, offset=None)
        self.vao.add_vbo(1, self.colors, ncomponents=3, stride=0, offset=None)
        self.vao.add_vbo(2, self.normals, ncomponents=3, stride=0, offset=None)
        self.vao.add_vbo(3, self.tex_coords, ncomponents=2, stride=0, offset=None)
        self.vao.add_ebo(self.indices)
        
        # Activate shader program and set transformation matrices
        GL.glUseProgram(self.shader.render_idx)

        self.uma.setup_texture("texture1", "texture/sky.jpeg")

        # Set up model, view, projection matrices (initialize as identity matrices)
        self.model = glm.mat4(1.0)
        self.view = glm.lookAt(glm.vec3(0, 0, 1), glm.vec3(0, 0, 0), glm.vec3(0, 1, 0))
        self.projection = glm.perspective(glm.radians(45.0), 800.0 / 600.0, 0.1, 100.0)

        model_view_matrix = self.view * self.model 
        self.uma.upload_uniform_matrix4fv(np.array(model_view_matrix), 'modelview', True)
        self.uma.upload_uniform_matrix4fv(np.array(self.projection), 'projection', True)

        # Light setup
        I_light = np.array([
            [0.9, 0.4, 0.6],  # diffuse
            [0.9, 0.4, 0.6],  # specular
            [0.9, 0.4, 0.6],  # ambient
        ], dtype=np.float32)
        light_pos = np.array([0, 0.5, 0.9], dtype=np.float32)

        self.uma.upload_uniform_matrix3fv(I_light, 'I_light', False)
        self.uma.upload_uniform_vector3fv(light_pos, 'light_pos')

        # Materials setup
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

        # Update transformation matrices if they have changed
        model_view_matrix = self.view * self.model 
        self.uma.upload_uniform_matrix4fv(np.array(model_view_matrix, dtype=np.float32), 'modelview', True)
        self.uma.upload_uniform_matrix4fv(np.array(self.projection, dtype=np.float32), 'projection', True)

        # Activate VAO and draw the pyramid
        # Draw base with texture
        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
        self.vao.activate()
        GL.glDrawElements(GL.GL_TRIANGLE_FAN, 5, GL.GL_UNSIGNED_INT, None)

        # Draw sides in polygon line mode
        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_LINE)
        GL.glDrawElements(GL.GL_TRIANGLES, self.indices.shape[0] - 5, GL.GL_UNSIGNED_INT, ctypes.c_void_p(5 * 4))

