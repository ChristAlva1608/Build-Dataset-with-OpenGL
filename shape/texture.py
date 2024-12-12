from OpenGL import GL
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt

class Texture:
    def __init__(self, shader, sampler_name, texture_unit, path):
        self.shader = shader
        self.texture_id = GL.glGenTextures(1)
        self.sampler_name = sampler_name
        self.texture_unit = GL.GL_TEXTURE0 + texture_unit
        self.path = path

    def load(self):
        try:
            image = Image.open(self.path)
            img_data = np.array(image)

            print(f"Texture data shape: {img_data.shape}, dtype: {img_data.dtype}")

            # Convert to float32 and normalize if needed
            if img_data.dtype != np.float32:
                img_data = img_data.astype(np.float32) / 255.0
                print(img_data)

            # Determine format based on image channels
            if len(img_data.shape) == 2:  # Grayscale
                interformat = GL.GL_RED
                format = GL.GL_RED
            elif len(img_data.shape) == 3:  # RGB or RGBA
                if img_data.shape[2] == 3:  # RGB
                    interformat = GL.GL_RGB32F
                    format = GL.GL_RGB
                else:  # RGBA
                    interformat = GL.GL_RGBA32F
                    format = GL.GL_RGBA

            # Bind the texture
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture_id)
            GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, interformat, image.width, image.height,
                           0, format, GL.GL_FLOAT, img_data)
            GL.glGenerateMipmap(GL.GL_TEXTURE_2D)

            # Set texture parameters
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_REPEAT)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_REPEAT)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR_MIPMAP_LINEAR)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)

            # Unbind the texture
            GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

            print(f"Successfully loaded texture: {self.path}")
            return True

        except Exception as e:
            print(f"Failed to load texture {self.path}: {e}")
            return False

    def bind(self):
        GL.glActiveTexture(self.texture_unit)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture_id)

        location = GL.glGetUniformLocation(self.shader.render_idx, self.sampler_name)
        if location != -1:
            GL.glUniform1i(location, self.texture_unit - GL.GL_TEXTURE0)
        else:
            print(f"Uniform '{self.sampler_name}' not found in shader.")

    def unbind(self):
        GL.glActiveTexture(self.texture_unit)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)