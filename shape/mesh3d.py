from libs.shader import *
from libs.buffer import *
from libs import transform as T
from shape.texture import *

import glm
import numpy as np

class Mesh:
    def __init__(self, shader, vertices, normals, texcoords, indices, material):
        self.vao = VAO()
        self.shader = shader
        self.uma = UManager(self.shader)
        self.vertices = vertices
        self.normals = normals
        self.texcoords = texcoords
        self.indices = indices
        self.material = material

        # print('vertices: ', self.vertices[0])
        # print('normals: ', self.normals[0])
        # print('texcoords: ', self.texcoords[0])
        # print('indices: ', self.indices[0])
        # print('material: ', self.material)

    def update_shader(self, shader):
        self.shader = shader
        self.uma = UManager(self.shader)

    def setup(self):
        self.vao.add_vbo(0, self.vertices, ncomponents=3, stride=0, offset=None)
        if self.normals is not None:
            self.vao.add_vbo(1, self.normals, ncomponents=3, stride=0, offset=None)
        if self.texcoords is not None:
            self.vao.add_vbo(2, self.texcoords, ncomponents=2, stride=0, offset=None)

        self.vao.add_ebo(self.indices)

        GL.glUseProgram(self.shader.render_idx)

        self.model = glm.mat4(1.0)

        camera_pos = glm.vec3(0.0, 0.0, 0.0)
        camera_target = glm.vec3(0.0, 0.0, 0.0)
        up_vector = glm.vec3(0.0, 1.0, 0.0)

        self.view = glm.lookAt(camera_pos, camera_target, up_vector)
        self.projection = glm.perspective(glm.radians(45.0), 1400.0 / 700.0, 0.1, 100.0)

        self.uma.upload_uniform_matrix4fv(np.array(self.model, dtype=np.float32), 'model', True)
        self.uma.upload_uniform_matrix4fv(np.array(self.view, dtype=np.float32), 'view', True)
        self.uma.upload_uniform_matrix4fv(np.array(self.projection), 'projection', True)

        # Setup texture
        try:
            # self.uma.setup_texture('texture_diffuse', 'obj/house_interior/double_sopha_wood_base_bace_uv.jpg')
            # print(f"Successfully loaded texture diffuse: 'obj/house_interior/double_sopha_wood_base_bace_uv.jpg'")
            if self.material['map_Kd']:
                texture_path = self.material['map_Kd'].find()
                if os.path.exists(texture_path):
                    self.uma.setup_texture('texture_diffuse', texture_path)
                    print(f"Successfully loaded texture diffuse: {texture_path}")
                    # self.diffuse_texture = Texture(self.shader, 'texture_diffuse', 0, texture_path)
                    # self.diffuse_texture.load()

            # if self.material['map_Ks']:
            #     texture_path = self.material['map_Ks'].find()
            #     if os.path.exists(texture_path):
            #         self.uma.setup_texture('texture_specular', texture_path)
            #         print(f"Successfully loaded texture specular: {texture_path}")
            #         # self.specular_texture = Texture(self.shader, 'texture_specular', 1, texture_path)
            #         # self.specular_texture.load()

            if self.material['map_Ka']:
                texture_path = self.material['map_Ka'].find()
                if os.path.exists(texture_path):
                    self.uma.setup_texture('texture_ambient', texture_path)
                    print(f"Successfully loaded texture ambient: {texture_path}")
                    # self.ambient_texture = Texture(self.shader, 'texture_ambient', 1, texture_path)
                    # self.ambient_texture.load()

            if self.material['map_bump']:
                texture_path = self.material['map_bump'].find()
                if os.path.exists(texture_path):
                    self.uma.setup_texture('texture_bump', texture_path)
                    print(f"Successfully loaded texture bump: {texture_path}")
                    # self.bump_texture = Texture(self.shader, 'texture_bump', 2, texture_path)
                    # self.bump_texture.load()
        except Exception as e:
            print(f"Error loading textures: {e}")

        I_light = np.array([
            [0.9, 0.4, 0.6],
            [0.9, 0.4, 0.6],
            [0.9, 0.4, 0.6],
        ], dtype=np.float32)
        light_pos = np.array([0, 0.5, 0.9], dtype=np.float32)

        self.uma.upload_uniform_matrix3fv(I_light, 'I_light', False)
        self.uma.upload_uniform_vector3fv(light_pos, 'light_pos')

        # self.K_materials = np.array([
        #     self.material['Kd'],
        #     self.material['Ks'],
        #     self.material['Ka'],
        # ], dtype=np.float32)

        # self.shininess = self.material['Ns']

        self.K_materials = np.array([
            [0.6, 0.4, 0.7],
            [0.6, 0.4, 0.7],
            [0.6, 0.4, 0.7],
        ], dtype=np.float32)
        self.shininess = 100.0

        phong_factor = 0.8
 
        self.uma.upload_uniform_matrix3fv(self.K_materials, 'K_materials', False)
        self.uma.upload_uniform_scalar1f(self.shininess, 'shininess')
        # self.uma.upload_uniform_scalar1i(1, 'mode')
        self.uma.upload_uniform_scalar1f(phong_factor, 'phong_factor')

        return self

    def draw(self, model, view, projection):
        GL.glUseProgram(self.shader.render_idx)

        # self.uma.upload_uniform_matrix3fv(self.K_materials, 'K_materials', False)
        # self.uma.upload_uniform_scalar1f(self.shininess, 'shininess')

        self.uma.upload_uniform_matrix4fv(np.array(model, dtype=np.float32), 'model', True)
        self.uma.upload_uniform_matrix4fv(np.array(view, dtype=np.float32), 'view', True)
        self.uma.upload_uniform_matrix4fv(np.array(projection, dtype=np.float32), 'projection', True)
        
        self.vao.activate()

        GL.glDrawElements(GL.GL_TRIANGLES, len(self.indices), GL.GL_UNSIGNED_INT, None)