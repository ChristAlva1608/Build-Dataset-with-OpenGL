from libs.shader import *
from libs.buffer import *
from libs import transform as T
import glm
import numpy as np
import trimesh

class Obj:
    def __init__(self, vert_shader, frag_shader, file_path):
        self.vao = VAO()
        self.vertices, self.normals, self.indices = self.load_obj(file_path)
        self.shader = Shader(vert_shader, frag_shader)
        self.uma = UManager(self.shader)
    
    def load_obj(self, file_path):
        scene = trimesh.load(file_path)
        if isinstance(scene,trimesh.Scene):
            scene = scene.to_mesh()
        vertices = scene.vertices.astype(np.float32)
        normals = scene.vertex_normals.astype(np.float32)
        indices = scene.faces.astype(np.uint32)
        if len(normals) == 0:
            return vertices, indices
        return vertices, normals, indices

    def load_texture(self, texture_path):
        self.uma.setup_texture(texture_path)

    def setup(self):
        self.vao.add_vbo(0, self.vertices, ncomponents=3, stride=0, offset=None)
        self.vao.add_vbo(2, self.normals, ncomponents=3, stride=0, offset=None)
        self.vao.add_ebo(self.indices)

        GL.glUseProgram(self.shader.render_idx)

        # Initial transformations
        self.model = glm.mat4(1.0)
        
        camera_pos = glm.vec3(0.0, 0.0, 5.0)
        camera_target = glm.vec3(0.0, 0.0, 0.0)
        up_vector = glm.vec3(0.0, 1.0, 0.0)
        
        self.view = glm.lookAt(camera_pos, camera_target, up_vector)
        self.projection = glm.perspective(glm.radians(45.0), 800.0 / 600.0, 0.1, 100.0)

        model_view_matrix = self.view * self.model
        
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

        GL.glDrawElements(GL.GL_TRIANGLES, len(self.indices)*3, GL.GL_UNSIGNED_INT, None)