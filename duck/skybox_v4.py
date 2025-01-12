import glfw
import numpy as np
from OpenGL.GL import *
import ctypes
from PIL import Image

from libs.shader import *
from libs.transform import *
from libs.buffer import *
from libs.camera import *
import glm
from itertools import cycle
import pyrr

faces = ['right.jpg','left.jpg', 'top.jpg', 'bottom.jpg', 'front.jpg', 'back.jpg']

class SkyBoxWithCube:
    def __init__(self, vert_default, frag_default, vert_skybox, frag_skybox, image_path, skybox_texture_dir):
        self.vertices_skybox = np.array([
            # Coordinates
            -1.0, -1.0,  1.0,  # 7
             1.0, -1.0,  1.0,  # 6
             1.0, -1.0, -1.0,  # 5
            -1.0, -1.0, -1.0,  # 4
            -1.0,  1.0,  1.0,  # 3
             1.0,  1.0,  1.0,  # 2
             1.0,  1.0, -1.0,  # 1
            -1.0,  1.0, -1.0   # 0
        ], dtype=np.float32)

        self.indices_skybox = np.array([
            # Right
            1, 2, 6,
            6, 5, 1,
            # Left
            0, 4, 7,
            7, 3, 0,
            # Top
            4, 5, 6,
            6, 7, 4,
            # Bottom
            0, 3, 2,
            2, 1, 0,
            # Back
            0, 1, 5,
            5, 4, 0,
            # Front
            3, 7, 6,
            6, 2, 3
        ], dtype=np.uint32)

        self.vertices_cube = np.array([
            [-0.5, -0.5, 0.5],
            [0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5],
            [-0.5, 0.5, 0.5],

            [-0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5],
            [0.5, 0.5, -0.5],
            [-0.5, 0.5, -0.5],

            [0.5, -0.5, -0.5],
            [0.5, 0.5, -0.5],
            [0.5, 0.5, 0.5],
            [0.5, -0.5, 0.5],

            [-0.5, 0.5, -0.5],
            [-0.5, -0.5, -0.5],
            [-0.5, -0.5, 0.5],
            [-0.5, 0.5, 0.5],

            [-0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5],
            [0.5, -0.5, 0.5],
            [-0.5, -0.5, 0.5],

            [0.5, 0.5, -0.5],
            [-0.5, 0.5, -0.5],
            [-0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5],
        ], dtype=np.float32)

        self.indices_cube = np.array([
            0,  1,  2,  2,  3,  0,
           4,  5,  6,  6,  7,  4,
           8,  9, 10, 10, 11,  8,
          12, 13, 14, 14, 15, 12,
          16, 17, 18, 18, 19, 16,
          20, 21, 22, 22, 23, 20], dtype=np.uint32)

        self.texcord_cube = np.array([
            # Front face
            [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0],  # Vertex 1, 2, 3, 4
            # Back face
            [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0],  # Vertex 5, 6, 7, 8
            # Top face
            [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0],  # Vertex 5, 6, 2, 1
            # Bottom face
            [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0],  # Vertex 8, 7, 3, 4
            # Left face
            [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0],  # Vertex 1, 4, 8, 5
            # Right face
            [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0],  # Vertex 2, 3, 7, 6
        ], dtype=np.float32)

        self.colors_cube = np.array([
            [0.583, 0.771, 0.014],
            [0.609, 0.115, 0.436],
            [0.327, 0.483, 0.844],
            [0.822, 0.569, 0.201],
            [0.310, 0.747, 0.185],
            [0.676, 0.977, 0.133],
            [0.559, 0.436, 0.730],
            [0.406, 0.615, 0.116],
        ], dtype=np.float32)

        self.vao_skybox = VAO()
        self.vao_cube = VAO()
        self.shader_skybox = Shader(vert_skybox, frag_skybox)
        self.shader_cube = Shader(vert_default, frag_default)
        self.uma_skybox = UManager(self.shader_skybox)
        self.uma_cube = UManager(self.shader_cube)

        # setup texture for skybox
        self.texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_CUBE_MAP, self.texture_id)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE)

        for i, face in enumerate(faces):
            face_path = skybox_texture_dir + face
            image = Image.open(face_path)
            # image = image.transpose(Image.FLIP_TOP_BOTTOM)  # Turn this off due to cubemap coordinate is not the same
            img_data = np.array(image, dtype=np.uint8)

            glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_RGB, image.width, image.height, 0, GL_RGB, GL_UNSIGNED_BYTE,
                         img_data)  # load to GPU

        glUseProgram(self.shader_skybox.render_idx)
        glUniform1i(glGetUniformLocation(self.shader_skybox.render_idx, "skybox"), 0)

        # setup texture for cube
        self.uma_cube.setup_texture("texture1", image_path)

    def setup(self):
        # setup for skybox
        self.vao_skybox.add_vbo(0, self.vertices_skybox, ncomponents=3, dtype=GL.GL_FLOAT, normalized=False, stride=0, offset=None)
        self.vao_skybox.add_ebo(self.indices_skybox)

        # setup for cube
        self.vao_cube.add_vbo(0, self.vertices_cube, ncomponents=3, dtype=GL.GL_FLOAT, normalized=False, stride=0, offset=None)
        self.vao_cube.add_vbo(1, self.colors_cube, ncomponents=3, dtype=GL.GL_FLOAT, normalized=False, stride=0, offset=None)
        # number 2 is for vertex normal
        self.vao_cube.add_vbo(3, self.texcord_cube, ncomponents=2, dtype=GL.GL_FLOAT, normalized=False, stride=0, offset=None)
        self.vao_cube.add_ebo(self.indices_cube)

        # Light setup (you can modify these values)
        I_light = np.array([
            [0.9, 0.4, 0.6],  # diffuse
            [0.9, 0.4, 0.6],  # specular
            [0.9, 0.4, 0.6],  # ambient
        ], dtype=np.float32)
        light_pos = np.array([0, 0.5, 0.9], dtype=np.float32)

        # Materials setup (you can modify these values)
        K_materials = np.array([
            [0.6, 0.4, 0.7],  # diffuse
            [0.6, 0.4, 0.7],  # specular
            [0.6, 0.4, 0.7],  # ambient
        ], dtype=np.float32)
        shininess = 100.0

        self.uma_cube.upload_uniform_matrix3fv(I_light, 'I_light', False)
        self.uma_cube.upload_uniform_vector3fv(light_pos, 'light_pos')
        self.uma_cube.upload_uniform_matrix3fv(K_materials, 'K_materials', False)
        self.uma_cube.upload_uniform_scalar1f(shininess, 'shininess')

        return self

    def draw(self, model, view, skybox_view, projection):

        self.vao_cube.activate()
        glUseProgram(self.shader_cube.render_idx)

        self.uma_cube.upload_uniform_matrix4fv(np.array(model), 'model', True)
        self.uma_cube.upload_uniform_matrix4fv(np.array(view), 'view', True)
        self.uma_cube.upload_uniform_matrix4fv(np.array(projection), 'projection', True)
        glDrawElements(GL_TRIANGLES, len(self.indices_cube), GL_UNSIGNED_INT, None)


        self.vao_skybox.activate()
        glDepthFunc(GL_LEQUAL)

        glUseProgram(self.shader_skybox.render_idx)

        self.uma_skybox.upload_uniform_matrix4fv(np.array(model), 'model', True)
        self.uma_skybox.upload_uniform_matrix4fv(np.array(skybox_view), 'view', True)
        self.uma_skybox.upload_uniform_matrix4fv(np.array(projection), 'projection', True)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_CUBE_MAP, self.texture_id)
        glDrawElements(GL_TRIANGLES, len(self.indices_skybox), GL_UNSIGNED_INT, None)

        glDepthFunc(GL_LESS) # switch back to normal depth function
        # self.vao_skybox.deactivate()


class Viewer:
    """ GLFW viewer windows, with classic initialization & graphics loop """
    def __init__(self, width=800, height=600):
        self.fill_modes = cycle([GL.GL_LINE, GL.GL_POINT, GL.GL_FILL])

        # version hints: create GL windows with >= OpenGL 3.3 and core profile
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL.GL_TRUE)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.RESIZABLE, False)
        self.win = glfw.create_window(width, height, 'Viewer', None, None)

        # Make the window's context current
        glfw.make_context_current(self.win)

        # Enable depth testing
        GL.glEnable(GL.GL_DEPTH_TEST)

        # Initialize trackball
        self.trackball = Trackball()
        self.mouse = (0, 0)

        self.cameraSpeed = 0.05
        self.cameraPos = glm.vec3(0.0, 0.0, 1.0)
        self.cameraFront = glm.vec3(0.0, 0.0, -1.0)
        self.cameraUp = glm.vec3(0.0, 1.0, 0.0)

        glfw.set_key_callback(self.win, self.on_key)
        glfw.set_cursor_pos_callback(self.win, self.on_mouse_move)
        glfw.set_scroll_callback(self.win, self.on_scroll)

        # useful message to check OpenGL renderer characteristics
        print('OpenGL', GL.glGetString(GL.GL_VERSION).decode() + ', GLSL',
              GL.glGetString(GL.GL_SHADING_LANGUAGE_VERSION).decode() +
              ', Renderer', GL.glGetString(GL.GL_RENDERER).decode())

        # initialize GL by setting viewport and default render characteristics
        GL.glClearColor(0.2, 0.3, 0.3, 1.0)
        self.drawables = []

    def on_mouse_move(self, win, xpos, ypos):
        """ Rotate on left-click & drag, pan on right-click & drag """
        old = self.mouse
        self.mouse = (xpos, glfw.get_window_size(win)[1] - ypos)
        if glfw.get_mouse_button(win, glfw.MOUSE_BUTTON_LEFT):
            self.trackball.drag(old, self.mouse, glfw.get_window_size(win))
        if glfw.get_mouse_button(win, glfw.MOUSE_BUTTON_RIGHT):
            self.trackball.pan(old, self.mouse)

    def on_scroll(self, win, _deltax, deltay):
        """ Scroll controls the camera distance to trackball center """
        self.trackball.zoom(deltay, glfw.get_window_size(win)[1])

    def on_key(self, _win, key, _scancode, action, _mods):
        if action == glfw.PRESS or action == glfw.REPEAT:
            if key == glfw.KEY_ESCAPE or key == glfw.KEY_Q:
                glfw.set_window_should_close(self.win, True)

            if key == glfw.KEY_J:
                GL.glPolygonMode(GL.GL_FRONT_AND_BACK, next(self.fill_modes))

            if key == glfw.KEY_W:
                self.cameraPos += self.cameraSpeed * self.cameraFront
            if key == glfw.KEY_S:
                self.cameraPos -= self.cameraSpeed * self.cameraFront
            if key == glfw.KEY_A:
                self.cameraPos -= glm.normalize(glm.cross(self.cameraFront, self.cameraUp)) * self.cameraSpeed
            if key == glfw.KEY_D:
                self.cameraPos += glm.normalize(glm.cross(self.cameraFront, self.cameraUp)) * self.cameraSpeed
            if key == glfw.KEY_SPACE:  # Move up
                self.cameraPos += self.cameraSpeed * self.cameraUp
            if key == glfw.KEY_LEFT_SHIFT:  # Move down
                self.cameraPos -= self.cameraSpeed * self.cameraUp

            for drawable in self.drawables:
                if hasattr(drawable, 'key_handler'):
                    drawable.key_handler(key)

    def run(self):
        """ Main render loop for this OpenGL windows """
        while not glfw.window_should_close(self.win):
            # clear draw buffer
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            win_size = glfw.get_window_size(self.win)

            self.model = glm.mat4(1.0)
            self.view = self.trackball.view_matrix2(self.cameraPos)
            self.skybox_view = self.trackball.skybox_view_matrix()
            self.projection = self.trackball.projection_matrix(win_size)

            # draw our scene objects
            for drawable in self.drawables:
                drawable.draw(self.model, self.view, self.skybox_view, self.projection)

            glfw.swap_buffers(self.win)
            # Poll for and process events
            glfw.poll_events()

    def add(self, *drawables):
        """ add objects to draw in this windows """
        self.drawables.extend(drawables)

default_vertex_shader = "./shader/phong_duck.vert"
default_fragment_shader = "./shader/phong_duck.frag"
skybox_vertex_shader = "./shader/skybox.vert"
skybox_fragment_shader = "./shader/skybox.frag"
texture_cube = "./image/ledinh.jpeg"
skybox_image_dir = "./image/skybox_1/"

# start from here
def main():
    viewer = Viewer()
    model = SkyBoxWithCube(
        default_vertex_shader,
        default_fragment_shader,
        skybox_vertex_shader,
        skybox_fragment_shader,
        texture_cube,
        skybox_image_dir)
    model.setup()
    viewer.add(model)
    viewer.run()

glfw.init()
main()
glfw.terminate()


