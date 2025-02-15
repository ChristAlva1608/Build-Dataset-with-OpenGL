from shape.scene3D_v3 import *
from libs.shader import *
from libs.transform import *
from libs.buffer import *
from libs.camera import *

class Cube:
    def __init__(self, vert_shader, frag_shader):
        self.vertices = np.array([
            [-0.5, -0.5, 0.5],  # Vertex 0
            [0.5, -0.5, 0.5],  # Vertex 1
            [0.5, 0.5, 0.5],  # Vertex 2
            [-0.5, 0.5, 0.5],  # Vertex 3
            [-0.5, -0.5, -0.5],  # Vertex 4
            [0.5, -0.5, -0.5],  # Vertex 5
            [0.5, 0.5, -0.5],  # Vertex 6
            [-0.5, 0.5, -0.5],  # Vertex 7
        ], dtype=np.float32)

        # Unique colors (one for each vertex)
        self.colors = np.array([
            [0.583, 0.771, 0.014],
            [0.609, 0.115, 0.436],
            [0.327, 0.483, 0.844],
            [0.822, 0.569, 0.201],
            [0.310, 0.747, 0.185],
            [0.676, 0.977, 0.133],
            [0.559, 0.436, 0.730],
            [0.406, 0.615, 0.116],
        ], dtype=np.float32)

        self.indices = np.array([
            0, 1, 2,
            2, 3, 0,
            4, 5, 6,
            6, 7, 4,
            4, 5, 1,
            1, 0, 4,
            6, 7, 3,
            3, 2, 6,
            5, 6, 2,
            2, 1, 5,
            7, 4, 0,
            0, 3, 7
        ], dtype=np.uint32)

        self.vao = VAO()
        self.shader = Shader(vert_shader, frag_shader)
        self.uma = UManager(self.shader)

    def setup(self):
        self.vao.add_vbo(0, self.vertices, ncomponents=3, dtype=GL.GL_FLOAT, normalized=False, stride=0, offset=None)
        # self.vao.add_vbo(1, self.colors, ncomponents=3, dtype=GL.GL_FLOAT, normalized=False, stride=0, offset=None)
        self.vao.add_ebo(self.indices)

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

    def draw(self, model, view, projection):
        self.vao.activate()
        glUseProgram(self.shader.render_idx)

        if (model is not None):
            self.model = model
        if (view is not None):
            self.view = view
        if (projection is not None):
            self.projection = projection

        self.uma.upload_uniform_matrix4fv(np.array(self.model), 'model', True)
        self.uma.upload_uniform_matrix4fv(np.array(self.view), 'view', True)
        self.uma.upload_uniform_matrix4fv(np.array(self.projection), 'projection', True)

        # glDrawArrays(GL.GL_TRIANGLES, 0, 12*3)
        glDrawElements(GL_TRIANGLES, len(self.indices), GL_UNSIGNED_INT, None)

class Viewer:
    """ GLFW viewer windows, with classic initialization & graphics loop """
    def __init__(self, width, height):
        self.width = width
        self.height = height
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
        GL.glDepthFunc(GL.GL_LESS)

        # trackball parameters
        self.trackball = Trackball()
        self.mouse = (0, 0)

        # camera parameters
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
            print("Win size type", type(win_size))

            # draw our scene objects
            for drawable in self.drawables:
                drawable.set_mode(1)

                model = glm.mat4(1.0)
                # view = self.trackball.view_matrix3(self.cameraPos, self.cameraFront, self.cameraUp)
                view = self.trackball.view_matrix()
                projection = self.trackball.projection_matrix(win_size)
                drawable.update_attribute('model_matrix', model)
                drawable.update_attribute('view_matrix', view)
                drawable.update_attribute('projection_matrix', projection)

                drawable.draw(self.cameraPos)

            glfw.swap_buffers(self.win)
            # Poll for and process events
            glfw.poll_events()

    def add(self, *drawables):
        """ add objects to draw in this windows """
        self.drawables.extend(drawables)

obj_file = "obj/floor/house_interior.obj"

# obj_file = "obj/house_interior/house_interior.obj"
# obj_file = "obj/warehouse/ware-house.obj"

def main():
    viewer = Viewer(1200, 700)
    shader = Shader('shader/depth_texture.vert', 'shader/depth_texture.frag')
    scene = Scene(shader, obj_file)
    scene.setup()
    viewer.add(scene)

    viewer.run()

glfw.init()
main()
glfw.terminate()
