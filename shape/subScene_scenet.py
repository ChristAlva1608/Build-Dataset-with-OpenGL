from libs.shader import *
from libs.buffer import *
from libs import transform as T
from shape.texture import *
from enum import Enum
import os

import glm
import numpy as np
from OpenGL.GL import *

class TextureMap(Enum):
    KA = ("map_Ka", "texture_ambient")
    KD = ("map_Kd", "texture_diffuse")
    KS = ("map_Ks", "texture_specular")
    REFL = ("map_refl", "texture_refl")
    BUMP = ("map_bump", "texture_bump")

class SubScene:
    def __init__(self, shader, vert, teco, normals, material, dir_path):
        self.shader = shader
        self.vao = VAO()
        self.uma = UManager(self.shader)

        self.use_texture = True if (len(teco) > 0 and teco[0] != -1 and material is not None) else False

        # init vertex attributes
        self.vertices = np.array(vert, dtype=np.float32)
        self.textcoords = np.array(teco, dtype=np.float32)
        self.normals = np.array(normals, dtype=np.float32)

        # init materials
        self.materials = material

        # init transformation matrix
        self.model = glm.mat4(1.0)
        self.view = glm.lookAt(glm.vec3(0, 0, 0), glm.vec3(0, 0, 0), glm.vec3(0, 1, 0))
        self.projection = glm.perspective(glm.radians(45.0), 800.0 / 600.0, 0.1, 100.0)

        # init texture
        self.texture_id = {}
        self.texture_flags = {}

        texture_found = False
        if self.use_texture:
            for texture in TextureMap:
                map_key, uniform_name = texture.value
                self.texture_flags[map_key] = False  # Initialize texture flag as False
                texture_path = self.materials.get(map_key)

                if texture_path:
                    if map_key == 'map_bump':
                        texture_path = texture_path['filename']
                    self.texture_path = os.path.join(dir_path, texture_path)

                    if os.path.exists(self.texture_path):
                        self.texture_flags[map_key] = True
                        self.texture_id[map_key] = self.uma.setup_texture(uniform_name, self.texture_path)
                        texture_found = True
                    else:
                        print("Missing texture", self.texture_path)

        if not texture_found:
            self.use_texture = False

    def get_model_matrix(self):
        return self.model

    def get_view_matrix(self):
        return self.view
    
    def get_projection_matrix(self):
        return self.projection
    
    def update_shader(self, shader):
        self.shader = shader
        self.uma = UManager(self.shader)

    def update_lightPos(self, lightPos):
        self.uma.upload_uniform_vector3fv(np.array(lightPos), "lightPos")

    def update_lightColor(self, lightColor):
        self.uma.upload_uniform_vector3fv(np.array(lightColor), "lightColor")

    def update_shininess(self, shininess):
        self.uma.upload_uniform_scalar1f(shininess, 'shininess')

    def update_model_matrix(self, model):
        self.model = model

    def update_view_matrix(self, view):
        self.view = view

    def update_projection_matrix(self, projection):
        self.projection = projection

    def update_texture(self, texture_path):
        # Only update map_Kd, supposed an object only have 1 simple texture
        self.texture_id['map_Kd'] = self.uma.setup_texture("texture_diffuse", texture_path)

    def transform_vertices(self):
        transformed_vertices = []
        for vertex in self.vertices:
            # Convert to homogeneous coordinates
            if not isinstance(self.model, glm.mat4):
                self.model = glm.mat4(*self.model.flatten())
            if not isinstance(self.view, glm.mat4):
                self.view = glm.mat4(*self.view.flatten())
            if not isinstance(self.projection, glm.mat4):
                self.projection = glm.mat4(*self.projection.flatten())
            vertex_homogeneous = glm.vec4(vertex[0], vertex[1], vertex[2], 1.0)

            # Apply model, view, and projection transformations
            vert_pos4 = self.projection * self.view * self.model * vertex_homogeneous
            # vert_pos = glm.vec3(vert_pos4) / vert_pos4.w  # Perspective divide

            # Append the transformed position and normal to the results
            transformed_vertices.append(vert_pos4)

        return transformed_vertices

    def setup(self):
        self.vao.add_vbo(0, self.vertices, ncomponents=3, dtype=GL.GL_FLOAT, normalized=False, stride=0, offset=None)
        self.vao.add_vbo(1, self.normals, ncomponents=3, dtype=GL.GL_FLOAT, normalized=False, stride=0, offset=None)
        self.vao.add_vbo(2, self.textcoords, ncomponents=2, dtype=GL.GL_FLOAT, normalized=False, stride=0, offset=None)

        light_pos = glm.vec3(250, 250, 300)
        light_color = glm.vec3(1.0, 1.0, 1.0) # only affect the current object, not the light source

        # lighting setup
        self.uma.upload_uniform_vector3fv(np.array(light_pos), "lightPos")
        self.uma.upload_uniform_vector3fv(np.array(light_color), "lightColor")

        if self.materials:
            self.diffuse = self.materials.get('Kd') if self.materials.get('Kd') is not None else [0.5, 0.5, 0.5]
            self.specular = self.materials.get('Ks') if self.materials.get('Ks') is not None else [0.5, 0.5, 0.5]
            self.ambient = self.materials.get('Ka') if self.materials.get('Ka') is not None else [0.5, 0.5, 0.5]
            self.shininess = self.materials.get('Ns') if self.materials.get('Ns') is not None else 100.0
        else:
            self.diffuse = [0.5, 0.5, 0.5]
            self.specular = [0.5, 0.5, 0.5]
            self.ambient = [0.5, 0.5, 0.5]
            self.shininess = 100.0
        return self

    def draw(self, cameraPos):
        self.vao.activate()
        glUseProgram(self.shader.render_idx)

        if self.use_texture: # use texture
            self.uma.upload_uniform_scalar1i(1, "use_texture")
        else:
            self.uma.upload_uniform_scalar1i(0, "use_texture")

        object_color = glm.vec3(0.5, 0.5, 0.5)
        self.uma.upload_uniform_vector3fv(np.array(object_color), "objectColor")

        self.uma.upload_uniform_vector3fv(np.array(self.diffuse), 'diffuseStrength')
        self.uma.upload_uniform_vector3fv(np.array(self.specular), 'specularStrength')
        self.uma.upload_uniform_vector3fv(np.array(self.ambient), 'ambientStrength')
        self.uma.upload_uniform_scalar1f(self.shininess, 'shininess')

        # Bind and upload textures dynamically
        if self.use_texture:
            for idx, texture in enumerate(TextureMap):
                map_key, uniform_name = texture.value
                if self.texture_flags[map_key]:
                    glActiveTexture(GL_TEXTURE0 + idx)
                    glBindTexture(GL_TEXTURE_2D, self.texture_id[map_key])
                    glUniform1i(glGetUniformLocation(self.shader.render_idx, uniform_name), idx)

        self.uma.upload_uniform_vector3fv(np.array(cameraPos), "viewPos")

        self.uma.upload_uniform_matrix4fv(np.array(self.model), 'model', True)
        self.uma.upload_uniform_matrix4fv(np.array(self.view), 'view', True)
        self.uma.upload_uniform_matrix4fv(np.array(self.projection), 'projection', True)

        glDrawArrays(GL.GL_TRIANGLES, 0, len(self.vertices))

        # Unbind all textures
        if self.use_texture:
            for idx, texture in enumerate(TextureMap):
                glActiveTexture(GL_TEXTURE0 + idx)
                glBindTexture(GL_TEXTURE_2D, 0)

        self.vao.deactivate()