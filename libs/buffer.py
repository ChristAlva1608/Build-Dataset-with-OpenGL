from .shader import *
import OpenGL.GL as GL
import cv2
from PIL import Image


class VAO(object):
    def __init__(self):
        self.vao = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self.vao)
        GL.glBindVertexArray(0)
        self.vbo = {}
        self.ebo = None

    def add_vbo(self, location, data,
               ncomponents=3, dtype=GL.GL_FLOAT, normalized=False, stride=0, offset=None):
        self.activate()
        buffer_idx = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, buffer_idx)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, data, GL.GL_STATIC_DRAW)
        #location = GL.glGetAttribLocation(self.shader.render_idx, name)
        GL.glVertexAttribPointer(location, ncomponents, dtype, normalized, stride, offset)
        GL.glEnableVertexAttribArray(location)
        self.vbo[location] = buffer_idx
        self.deactivate()

    def add_ebo(self, indices):
        self.activate()
        self.ebo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, indices, GL.GL_STATIC_DRAW)
        self.deactivate()

    def update_vbo(self, location, new_data):
        """
        Update VBO data at a given attribute location.
        Assumes the VBO has already been created.
        """
        if location not in self.vbo:
            raise ValueError(f"VBO at location {location} not found.")

        self.activate()
        buffer_idx = self.vbo[location]
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, buffer_idx)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, new_data.nbytes, new_data, GL.GL_STATIC_DRAW)
        # GL.glBufferSubData(GL.GL_ARRAY_BUFFER, offset=0, size=new_data.nbytes, data=new_data)
        self.deactivate()

    def __del__(self):
        GL.glDeleteVertexArrays(1, [self.vao])
        GL.glDeleteBuffers(1, list(self.vbo.values()))
        if self.ebo is not None:
            GL.glDeleteBuffers(1, [self.ebo])

    def activate(self):
        GL.glBindVertexArray(self.vao)  # activated

    def deactivate(self):
        GL.glBindVertexArray(0)  # activated


class UManager(object):
    def __init__(self, shader):
        self.shader = shader
        self.textures = {}

    @staticmethod
    def load_texture(filename):
        image = Image.open(filename)
        image = image.transpose(Image.FLIP_TOP_BOTTOM)  # Flip image for OpenGL
        img_mode = image.mode

        if img_mode == "I;16":
            img_data = np.array(image, dtype=np.uint16)  # Preserve 16-bit data
        elif img_mode == "P": # handle "palettised" mode up here due to openGl not support
            image = image.convert("RGB")
            img_data = np.array(image, dtype=np.uint8)
            img_mode = "RGB"
        else:
            img_data = np.array(image, dtype=np.uint8)
        return img_data, img_mode

    def _get_texture_loc(self):
        if not bool(self.textures):
            return 0
        else:
            locs = list(self.textures.keys())
            locs.sort(reverse=True)
            ret_id = locs[0] + 1
            return ret_id

    """
    * first call to setup_texture: activate GL.GL_TEXTURE0
        > use GL.glUniform1i to associate the activated texture to the texture in shading program (see fragment shader)
    * second call to setup_texture: activate GL.GL_TEXTURE1
        > use GL.glUniform1i to associate the activated texture to the texture in shading program (see fragment shader)
    * second call to setup_texture: activate GL.GL_TEXTURE2
        > use GL.glUniform1i to associate the activated texture to the texture in shading program (see fragment shader)
    and so on

    """
    def setup_texture(self, sampler_name, image_file):
        img_data, img_mode = UManager.load_texture(image_file)

        GL.glUseProgram(self.shader.render_idx)  # must call before calling to GL.glUniform1i
        texture_idx = GL.glGenTextures(1)
        binding_loc = self._get_texture_loc()
        self.textures[binding_loc] = {}
        self.textures[binding_loc]["id"] = texture_idx
        self.textures[binding_loc]["name"] = sampler_name

        # Determine optimal unpack alignment (default width should divided by 4)
        if len(img_data.shape) == 3:
            _, width, num_channels = img_data.shape
            row_size = width * num_channels
        elif len(img_data.shape) == 2:
            row_size = img_data.shape[1]
        else:
            row_size = img_data.shape[0]  # may need handle in the future
        alignment = 4 if (row_size % 4 == 0) else 1
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, alignment)

        GL.glActiveTexture(GL.GL_TEXTURE0 + binding_loc)  # activate texture GL.GL_TEXTURE0, GL.GL_TEXTURE1, ...
        GL.glBindTexture(GL.GL_TEXTURE_2D, texture_idx)
        GL.glUniform1i(GL.glGetUniformLocation(self.shader.render_idx, sampler_name),
                       binding_loc)

        if img_mode == 'RGB':
            GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGB,
                            img_data.shape[1], img_data.shape[0],
                            0, GL.GL_RGB, GL.GL_UNSIGNED_BYTE, img_data)
        elif img_mode == 'RGBA':
            GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA,
                            img_data.shape[1], img_data.shape[0],
                            0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, img_data)
        elif img_mode == 'LA':
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_SWIZZLE_R, GL.GL_RED)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_SWIZZLE_G, GL.GL_RED)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_SWIZZLE_B, GL.GL_RED)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_SWIZZLE_A, GL.GL_GREEN)
            GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RG, img_data.shape[1],
                            img_data.shape[0], 0, GL.GL_RG, GL.GL_UNSIGNED_BYTE, img_data)
        elif img_mode == 'L':
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_SWIZZLE_G, GL.GL_RED)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_SWIZZLE_B, GL.GL_RED)
            GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RED, img_data.shape[1],
                            img_data.shape[0], 0, GL.GL_RED, GL.GL_UNSIGNED_BYTE, img_data)
        elif img_mode == 'I;16':
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_SWIZZLE_G, GL.GL_RED)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_SWIZZLE_B, GL.GL_RED)
            GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_R16, img_data.shape[1],
                            img_data.shape[0], 0, GL.GL_RED, GL.GL_UNSIGNED_SHORT, img_data)
        else:
            print("Problem with new image mode", img_mode)
            print("Image file", image_file)

        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)

        return texture_idx

    def upload_uniform_matrix4fv(self, matrix, name, transpose=True):
        GL.glUseProgram(self.shader.render_idx)
        location = GL.glGetUniformLocation(self.shader.render_idx, name)
        GL.glUniformMatrix4fv(location, 1, transpose, matrix)

    def upload_uniform_matrix3fv(self, matrix, name, transpose=False):
        GL.glUseProgram(self.shader.render_idx)
        location = GL.glGetUniformLocation(self.shader.render_idx, name)
        GL.glUniformMatrix3fv(location, 1, transpose, matrix)

    def upload_uniform_vector4fv(self, vector, name):
        GL.glUseProgram(self.shader.render_idx)
        location = GL.glGetUniformLocation(self.shader.render_idx, name)
        GL.glUniform4fv(location, 1, vector)

    def upload_uniform_vector3fv(self, vector, name):
        GL.glUseProgram(self.shader.render_idx)
        location = GL.glGetUniformLocation(self.shader.render_idx, name)
        GL.glUniform3fv(location, 1, vector)

    def upload_uniform_scalar1f(self, scalar, name):
        GL.glUseProgram(self.shader.render_idx)
        location = GL.glGetUniformLocation(self.shader.render_idx, name)
        GL.glUniform1f(location, scalar)

    def upload_uniform_scalar1i(self, scalar, name):
        GL.glUseProgram(self.shader.render_idx)
        location = GL.glGetUniformLocation(self.shader.render_idx, name)
        GL.glUniform1i(location, scalar)