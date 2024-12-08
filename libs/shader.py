import OpenGL.GL as GL              # standard Python OpenGL wrapper
import numpy as np
import pandas as pd
import sys
import os


class Shader:
    """ Helper class to create and automatically destroy shader program """
    def __init__(self, vertex_source, fragment_source):
        """ Shader can be initialized with raw strings or source file names """
        self.render_idx = None
        vert = self._compile_shader(vertex_source, GL.GL_VERTEX_SHADER)
        frag = self._compile_shader(fragment_source, GL.GL_FRAGMENT_SHADER)
        if vert and frag:
            self.render_idx = GL.glCreateProgram()  
            GL.glAttachShader(self.render_idx, vert)
            GL.glAttachShader(self.render_idx, frag)
            GL.glLinkProgram(self.render_idx)
            GL.glDeleteShader(vert)
            GL.glDeleteShader(frag)
            status = GL.glGetProgramiv(self.render_idx, GL.GL_LINK_STATUS)
            if not status:
                print(GL.glGetProgramInfoLog(self.render_idx).decode('ascii'))
                sys.exit(1)

    def __del__(self):
        GL.glUseProgram(0)
        if self.render_idx:                      # if this is a valid shader object
            GL.glDeleteProgram(self.render_idx)  # object dies => destroy GL object

    def use(self):
        """ Activate the shader program """
        GL.glUseProgram(self.render_idx)

    def set_uniform(self, name, value):
        """ Set uniform value in shader """
        if self.render_idx is None:
            return

        # Activate the shader program
        GL.glUseProgram(self.render_idx)

        # Get uniform location
        location = GL.glGetUniformLocation(self.render_idx, name)
        
        # Check if uniform exists
        if location == -1:
            print(f"Warning: Uniform '{name}' not found in shader")
            return

        # Set uniform based on type
        if isinstance(value, float):
            GL.glUniform1f(location, value)
        elif isinstance(value, int):
            GL.glUniform1i(location, value)
        elif isinstance(value, tuple) or isinstance(value, list):
            if len(value) == 2:
                GL.glUniform2f(location, *value)
            elif len(value) == 3:
                GL.glUniform3f(location, *value)
            elif len(value) == 4:
                GL.glUniform4f(location, *value)
        elif isinstance(value, np.ndarray):
            if value.dtype == np.float32 and value.ndim == 1:
                if value.shape[0] == 16:  # 4x4 matrix
                    GL.glUniformMatrix4fv(location, 1, GL.GL_FALSE, value)
                elif value.shape[0] == 9:  # 3x3 matrix
                    GL.glUniformMatrix3fv(location, 1, GL.GL_FALSE, value)
                else:
                    print(f"Unsupported numpy array size for uniform '{name}'")
            else:
                print(f"Unsupported numpy array type for uniform '{name}'")

    @staticmethod
    def _compile_shader(src, shader_type):
        src = open(src, 'r').read() if os.path.exists(src) else src
        src = src.decode('ascii') if isinstance(src, bytes) else src
        shader = GL.glCreateShader(shader_type)
        GL.glShaderSource(shader, src)
        GL.glCompileShader(shader)
        status = GL.glGetShaderiv(shader, GL.GL_COMPILE_STATUS)
        src = ('%3d: %s' % (i + 1, l) for i, l in enumerate(src.splitlines()))
        if not status:
            log = GL.glGetShaderInfoLog(shader).decode('ascii')
            GL.glDeleteShader(shader)
            src = '\n'.join(src)
            print('Compile failed for %s\n%s\n%s' % (shader_type, log, src))
            sys.exit(1)
        return shader