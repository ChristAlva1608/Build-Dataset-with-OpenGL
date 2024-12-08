from OpenGL import GL
from PIL import Image
import numpy as np
import os

class Texture:
    def __init__(self):
        self.texture_id = GL.glGenTextures(1)
    
    def load(self, image):
        try:
            img_data = np.array(image)
            
            # Bind the texture
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture_id)
            
            # Set texture parameters
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_REPEAT)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_REPEAT)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR_MIPMAP_LINEAR)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
            
            # Determine format based on image channels
            if len(img_data.shape) == 2:  # Grayscale
                internal_format = GL.GL_RED
                format = GL.GL_RED
            elif len(img_data.shape) == 3:  # RGB or RGBA
                if img_data.shape[2] == 3:  # RGB
                    internal_format = GL.GL_RGB
                    format = GL.GL_RGB
                else:  # RGBA
                    internal_format = GL.GL_RGBA
                    format = GL.GL_RGBA
            
            # Upload texture data
            GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, internal_format, image.width, image.height,
                           0, format, GL.GL_UNSIGNED_BYTE, img_data)
            GL.glGenerateMipmap(GL.GL_TEXTURE_2D)
            
            return True
        except Exception as e:
            print(e)
            return False

    def bind(self, texture_unit=0):
        GL.glActiveTexture(GL.GL_TEXTURE0 + texture_unit)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture_id)

    def unbind(self):
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)