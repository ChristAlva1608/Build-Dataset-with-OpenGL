import OpenGL.GL as GL
import glfw
import numpy as np
import time
import random
import re
import glm
import trimesh
from itertools import cycle
import imgui
from imgui import Vec2
from imgui.integrations.glfw import GlfwRenderer
# from imgui.integrations.opengl import texture
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import io

from libs.buffer import *
from libs.camera import *
from libs.shader import *
from libs.transform import *

from object3D import *
# from scene3D import *
# from model3D import *
from quad import *
from vcamera import *
from sphere import *


PYTHONPATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, PYTHONPATH)

class Viewer:
    def __init__(self, width=1400, height=700):
        self.fill_modes = cycle([GL.GL_LINE, GL.GL_POINT, GL.GL_FILL])

        # GLFW initialization
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL.GL_TRUE)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.RESIZABLE, False)
        glfw.window_hint(glfw.DEPTH_BITS, 16)
        glfw.window_hint(glfw.DOUBLEBUFFER, True)
        glfw.window_hint(glfw.COCOA_RETINA_FRAMEBUFFER, False) # Turn off Retina scaling in MacOS

        self.win = glfw.create_window(width, height, 'Viewer', None, None)
        if not self.win:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")

        glfw.make_context_current(self.win)

        # Initialize window width and height
        self.win_width, self.win_height = width, height

        # Initialize imgui
        imgui.create_context()
        self.imgui_impl = GlfwRenderer(self.win)
        self.init_ui()

        # Enable depth testing
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDepthFunc(GL.GL_LESS)

        # Initialize shaders
        self.depth_shader = Shader("shader/depth.vert", "shader/depth.frag")
        self.main_shader = Shader("shader/phong.vert", "shader/phong.frag")
        self.phongex_shader = Shader("shader/phongex.vert", "shader/phongex.frag")
        self.texture_shader = Shader('shader/texture.vert', 'shader/texture.frag')

        # Initialize mouse parameters
        self.last_x = width / 2
        self.last_y = height / 2
        self.first_mouse = True
        self.left_mouse_pressed = False
        self.yaw = -90.0
        self.pitch = 0.0
        self.fov = 45.0

        # Initialize control flag
        self.diffuse_changed = False
        self.ambient_changed = False
        self.specular_changed = False
        self.diffuse = 0.0
        self.ambient = 0.0
        self.specular = 0.0

        # Initialize selection
        self.selected_obj = -1
        self.selected_scene = -1

        # Initialize trackball
        self.trackball = Trackball()
        self.mouse = (0, 0)

        # Initialize vcamera parameters
        self.cameraSpeed = 0.5
        self.cameraPos = glm.vec3(0.0, 0.0, 5.0)
        self.cameraFront = glm.vec3(0.0, 0.0, -1.0)
        self.cameraUp = glm.vec3(0.0, 1.0, 0.0)
        self.lastFrame = 0.0

        # Initialize for near & far planes
        self.near = 0
        self.far = 0
        self.near_colors = [0.0, 0.0, 0.0]
        self.far_colors = [1.0, 1.0, 1.0]

        # Initialize for camera
        self.camera1_cameraPos = glm.vec3(0.0, 0.0, 5.0)
        self.camera2_cameraPos = glm.vec3(5.0, 0.0, 0.0)
        self.camera1_cameraFront = glm.vec3(0.0, 0.0, -1.0)
        self.camera2_cameraFront = glm.vec3(1.0, 0.0, 0.0)

        # Initialize background color
        self.bg_colors = [0.2, 0.2, 0.2]
        self.bg_changed = False

        # Register callbacks
        glfw.set_key_callback(self.win, self.on_key)
        glfw.set_cursor_pos_callback(self.win, self.on_mouse_move)
        glfw.set_scroll_callback(self.win, self.scroll_callback)

        print('OpenGL', GL.glGetString(GL.GL_VERSION).decode() + ', GLSL',
              GL.glGetString(GL.GL_SHADING_LANGUAGE_VERSION).decode() +
              ', Renderer', GL.glGetString(GL.GL_RENDERER).decode())

        GL.glClearColor(1.0, 1.0, 1.0, 1.0)
        self.drawables = []

    def init_ui(self):
        self.font_size = 10
        self.load_font_size()

        self.scene_width = self.win_width // 7
        self.scene_height = self.win_height // 3

        self.material_config_width = self.scene_width
        self.material_config_height = self.scene_height

        self.depth_config_width = self.scene_width
        self.depth_config_height = self.scene_height

        self.operation_width = self.win_width // 7
        self.operation_height = self.win_height // 3

        self.command_width = self.win_width // 7
        self.command_height = self.win_height // 3

        self.rgb_view_width = (self.win_width - self.scene_width - self.operation_width - self.command_width) // 2
        self.rgb_view_height = self.win_height

        self.depth_view_width = self.rgb_view_width
        self.depth_view_height = self.rgb_view_height

    def save_rgb(self):

        # Create a numpy array to hold the pixel data
        pixels = GL.glReadPixels(0, 0, self.win_width, self.win_height*2, GL.GL_RGB, GL.GL_UNSIGNED_BYTE)

        # Convert the pixels into a numpy array
        rgb_image = np.frombuffer(pixels, dtype=np.uint8).reshape((int(self.win_height*2), int(self.win_width), 3))

        # Flip the image vertically (because OpenGL's origin is at the bottom-left corner)
        rgb_image = np.flipud(rgb_image)

        # Convert numpy array (or your image data format) to PIL Image
        rgb_image = Image.fromarray(rgb_image)

        # Create a unique file name using timestamp
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        file_name = f"rgb_image_{timestamp}.png"

        # Save the image to the local directory
        rgb_image.save(f"./rgb_images/{file_name}")
        print(f"Saved rgb image as {file_name}")

    def save_depth(self):

        # Create a numpy array to hold the pixel data
        pixels = GL.glReadPixels(self.win_width, 0, self.win_width, self.win_height*2, GL.GL_RGB, GL.GL_UNSIGNED_BYTE)

        # Convert the pixels into a numpy array
        depth_image = np.frombuffer(pixels, dtype=np.uint8).reshape((self.win_height*2, self.win_width, 3))

        # Flip the image vertically (because OpenGL's origin is at the bottom-left corner)
        depth_image = np.flipud(depth_image)

        # Convert numpy array (or your image data format) to PIL Image
        depth_image = Image.fromarray(depth_image)

        # Create a unique file name using timestamp
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        file_name = f"depth_image_{timestamp}.png"

        # Save the image to the local directory
        depth_image.save(f"./depth_images/{file_name}")
        print(f"Saved depth image as {file_name}")

    def get_yaw_pitch_from_direction(self, a, b):
        direction = glm.normalize(b - a)
        # Calculate yaw (angle in the XZ plane)
        yaw = glm.degrees(np.arctan2(direction.z, direction.x))
        # Calculate pitch (vertical angle)
        pitch = glm.degrees(np.arcsin(direction.y))
        return yaw, pitch

    # def texture_mapping(self):

    def load_texture(self, image_path):
        """Load an image file into a texture for ImGui"""
        image = Image.open(image_path)
        image = image.convert('RGBA')
        image_data = np.array(image)

        # Generate texture
        texture_id = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, texture_id)

        # Setup texture parameters
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)

        # Upload texture data
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA,
                        image.width, image.height, 0,
                        GL.GL_RGBA, GL.GL_UNSIGNED_BYTE,
                        image_data)

        return texture_id

    def button_with_icon(self, image_path, button_text, icon_size=(20, 20)):
        """Create a button with an icon and text"""
        icon_texture = self.load_texture(image_path)

        # Calculate total button width (icon + spacing + text)
        text_size = imgui.calc_text_size(button_text)
        button_width = icon_size[0] + 8 + text_size.x  # 8 pixels spacing
        button_height = max(icon_size[1], text_size.y)

        # Start button with custom size
        pressed = imgui.button("##hidden" + button_text, width=button_width, height=button_height)

        # Get button drawing position
        draw_list = imgui.get_window_draw_list()
        pos = imgui.get_item_rect_min()

        # Draw icon
        draw_list.add_image(
            icon_texture,
            (pos.x, pos.y),
            (pos.x + icon_size[0], pos.y + icon_size[1])
        )

        # Draw text
        text_pos_x = pos.x + icon_size[0] + 8  # 8 pixels spacing after icon
        text_pos_y = pos.y + (button_height - text_size.y) * 0.5  # Center text vertically
        draw_list.add_text(text_pos_x, text_pos_y, imgui.get_color_u32_rgba(1, 1, 1, 1), button_text)

        return pressed

    def run(self):
        while not glfw.window_should_close(self.win):
            GL.glClearColor(0.2, 0.2, 0.2, 1.0)

            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

            # Viewport for RGB Scene
            win_pos_width = self.scene_width + self.operation_width
            win_pos_height = self.win_height - self.rgb_view_height # start from bottom-left

            GL.glViewport(win_pos_width, win_pos_height, self.rgb_view_width, self.rgb_view_height)
            GL.glScissor(win_pos_width, win_pos_height, self.rgb_view_width, self.rgb_view_height)
            GL.glEnable(GL.GL_SCISSOR_TEST)
            GL.glClearColor(*self.bg_colors, 1.0)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT)

            GL.glUseProgram(self.texture_shader.render_idx)

            for drawable in self.drawables:
                drawable.update_shader(self.texture_shader)
                # drawable.setup()

                drawable.model = glm.mat4(1.0)
                drawable.view = self.trackball.view_matrix2(self.cameraPos)
                # drawable.view = self.trackball.view_matrix()
                drawable.projection = glm.perspective(glm.radians(self.fov), self.rgb_view_width / self.rgb_view_height, 0.1, 1000.0)

                # Normal rendering
                drawable.draw()

            # # Viewport for Depth Scene
            # win_pos_width = self.scene_width + self.operation_width + self.rgb_view_width
            # win_pos_height = self.win_height - self.depth_view_height # start from bottom-left
            # GL.glViewport(win_pos_width, win_pos_height, self.depth_view_width, self.depth_view_height)
            # GL.glScissor(win_pos_width, win_pos_height, self.depth_view_width, self.depth_view_height)
            # GL.glClearColor(*self.bg_colors, 1.0)
            # GL.glClear(GL.GL_COLOR_BUFFER_BIT)

            # GL.glUseProgram(self.depth_shader.render_idx)
            # for drawable in self.drawables:
            #     drawable.update_shader(self.depth_shader)
            #     # drawable.setup()

            #     # update color of depth map
            #     drawable.uma.upload_uniform_vector3fv(self.near_colors, 'nearColor')
            #     drawable.uma.upload_uniform_vector3fv(self.far_colors, 'farColor')

            #     drawable.model = glm.mat4(1.0)
            #     drawable.view = self.trackball.view_matrix2(self.cameraPos)
            #     # drawable.view = self.trackball.view_matrix()
            #     drawable.projection = glm.perspective(glm.radians(self.fov), self.depth_view_width / self.depth_view_height, 0.1, 1000.0)

            #     # Depth map rendering
            #     drawable.shader.set_uniform('near', 0.1)
            #     drawable.shader.set_uniform('far', 1000.0)
            #     drawable.draw()

            GL.glDisable(GL.GL_SCISSOR_TEST)

            self.imgui_menu()

            glfw.swap_buffers(self.win)
            glfw.poll_events()
            self.imgui_impl.process_inputs()

        self.imgui_impl.shutdown()

    def load_font_size(self):
        io = imgui.get_io()
        io.font_global_scale = 1

    def use_trackball(self):
        for drawable in self.drawables:
            win_size = glfw.get_window_size(self.win)
            drawable.view = self.trackball.view_matrix3()
            drawable.projection = self.trackball.projection_matrix(win_size)

    def move_camera_around(self):
        for drawable in self.drawables:
            drawable.model = glm.mat4(1.0)
            drawable.view = self.trackball.view_matrix2(self.cameraPos)
            drawable.projection = glm.perspective(glm.radians(self.fov), 800.0 / 600.0, 0.1, 100.0)

    def update_diffuse(self):
        for drawable in self.drawables:
            drawable.K_materials[0] = self.diffuse

    def update_specular(self):
        for drawable in self.drawables:
            drawable.K_materials[0] = self.specular

    def update_ambient(self):
        for drawable in self.drawables:
            drawable.K_materials[0] = self.ambient

    def imgui_menu(self):
        # Create a new frame
        imgui.new_frame()

        ########################################################################
        #                                Scene                                 #
        ########################################################################
        win_pos_width = 0
        win_pos_height = 0
        imgui.set_next_window_position(win_pos_width, win_pos_height)
        imgui.set_next_window_size(self.scene_width, self.scene_height)

        imgui.begin("Scene")

        # set item size
        imgui.set_window_font_scale(1)
        item_width = imgui.get_window_width() - 10 # padding

        _, self.selected_scene = imgui.combo(
            "Scene",
            int(self.selected_scene),
            ["House Interior", "Parking lot", "Bathroom"]
        )

        _, self.selected_obj = imgui.combo(
            "Object",
            int(self.selected_obj),
            ["Wuson", "Porsche", "Bowl"]
        )

        # imgui.set_next_item_width(imgui.get_window_width())
        camera1 = self.button_with_icon('icons/camera.png', 'Camera A')
        camera2 = self.button_with_icon('icons/camera.png', 'Camera B')

        if camera1:
            self.cameraPos = self.camera1_cameraPos
            self.cameraFront = self.camera1_cameraFront

        if camera2:
            self.cameraPos = self.camera2_cameraPos
            self.cameraFront = self.camera2_cameraFront

        # Adjust RGB
        self.bg_changed, self.bg_colors = imgui.input_float3('Color', self.bg_colors[0], self.bg_colors[1], self.bg_colors[2], format='%.2f')

        font_size = imgui.get_font_size()
        vertical_padding = 8
        button_height = font_size + vertical_padding*2
        imgui.set_cursor_pos((imgui.get_window_width()//4, imgui.get_window_height() - button_height))
        imgui.set_next_item_width(imgui.get_window_width()//2)
        if imgui.button("Load Model"):
            self.create_model()

        imgui.end()

        ########################################################################
        #                              Operation                              #
        ########################################################################
        win_pos_width = self.scene_width
        win_pos_height = 0
        imgui.set_next_window_position(win_pos_width, win_pos_height)
        imgui.set_next_window_size(self.operation_width, self.operation_height)
        imgui.begin("Operation")

        imgui.set_next_item_width(imgui.get_window_width())
        if imgui.button("Use Trackball"):
            self.use_trackball()

        imgui.set_next_item_width(imgui.get_window_width())
        if imgui.button("Move Camera"):
            self.move_camera_around()

        imgui.get_style().colors[imgui.COLOR_BUTTON_HOVERED] = imgui.Vec4(0.6, 1.0, 0.6, 1.0)  # Green hover color
        imgui.set_next_item_width(100)
        if imgui.button("Save all"):
            self.save_rgb()
            self.save_depth()

        imgui.same_line()
        imgui.set_next_item_width(100)
        if imgui.button("Save RGB"):
            self.save_rgb()

        imgui.same_line()
        imgui.set_next_item_width(100)
        if imgui.button("Save Depth Map"):
            self.save_depth()

        imgui.end()

        ########################################################################
        #                          Material Config                             #
        ########################################################################
        win_pos_width = 0
        win_pos_height = self.scene_height
        imgui.set_next_window_position(win_pos_width, win_pos_height)
        imgui.set_next_window_size(self.material_config_width, self.material_config_height)
        imgui.begin("Material Config")

        # Add Diffuse slider
        imgui.set_next_item_width(imgui.get_window_width()//2)
        self.diffuse_changed, diffuse_value = imgui.slider_float("Diffuse",
                                          self.diffuse,
                                          min_value=0.00,
                                          max_value=1,
                                          format="%.2f")
        if self.diffuse_changed:
            self.diffuse = diffuse_value
            self.update_diffuse()

        # Add Specular slider
        imgui.set_next_item_width(imgui.get_window_width()//2)
        self.specular_changed, specular_value = imgui.slider_float("Specular",
                                          self.specular,
                                          min_value=0.00,
                                          max_value=1,
                                          format="%.2f")
        if self.specular_changed:
            self.specular = specular_value
            self.update_specular()

        # Add Ambient slider
        imgui.set_next_item_width(imgui.get_window_width()//2)
        self.ambient_changed, ambient_value = imgui.slider_float("Ambient",
                                          self.ambient,
                                          min_value=0.00,
                                          max_value=1,
                                          format="%.2f")
        if self.ambient_changed:
            self.ambient = ambient_value
            self.update_ambient()

        imgui.end()

        ########################################################################
        #                              Depth Config                            #
        ########################################################################
        win_pos_width = 0
        win_pos_height = self.scene_height + self.material_config_height
        imgui.set_next_window_position(win_pos_width, win_pos_height)
        imgui.set_next_window_size(self.depth_config_width, self.depth_config_height)
        imgui.begin("Depth Config")

        # Add near plane slider
        # imgui.set_next_item_width(100)
        self.near_changed, near_value = imgui.slider_int("near",
                                          int(self.near),
                                          min_value=1,
                                          max_value=1000
                                          )
        if self.near_changed:
            self.near = near_value

        # Add far plane slider
        # imgui.set_next_item_width(100)
        self.far_changed, far_value = imgui.slider_int("far",
                                          int(self.far),
                                          min_value=1,
                                          max_value=1000
                                          )
        if self.far_changed:
            self.far = far_value

        # Adjust RGB
        self.near_color_changed, self.near_colors = imgui.input_float3('Near Color', self.near_colors[0], self.near_colors[1], self.near_colors[2], format='%.2f')
        self.far_color_changed, self.far_colors = imgui.input_float3('Far Color', self.far_colors[0], self.far_colors[1], self.far_colors[2], format='%.2f')

        imgui.end()

        ########################################################################
        #                               Command                                #
        ########################################################################
        win_pos_width = self.scene_width + self.operation_width + self.rgb_view_width + self.depth_view_width
        win_pos_height = 0
        imgui.set_next_window_position(win_pos_width, win_pos_height)
        imgui.set_next_window_size(self.command_width, self.command_width)
        imgui.begin("Command")

        imgui.end()

        imgui.render()
        self.imgui_impl.render(imgui.get_draw_data())

    def add(self, drawables):
        """ add objects to draw in this windows """
        self.drawables.extend(drawables)

    def on_key(self, _win, key, _scancode, action, _mods):
        """ 'Q' or 'Escape' quits """
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

            for drawable in self.drawables:
                if hasattr(drawable, 'key_handler'):
                    drawable.key_handler(key)

    def on_mouse_move(self, window, xpos, ypos):
        """ Rotate on left-click & drag, pan on right-click & drag """
        old = self.mouse
        self.mouse = (xpos, glfw.get_window_size(window)[1] - ypos)
        if glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT):
            self.trackball.drag(old, self.mouse, glfw.get_window_size(window))

        if glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT):
            self.trackball.pan(old, self.mouse)

    def scroll_callback(self, window, xoffset, yoffset):
        self.fov -= float(yoffset)
        self.trackball.zoom(yoffset, glfw.get_window_size(window)[1])

    def create_model(self):
        model = []
        self.drawables.clear()

        # Create Scene model
        scene_files = {
            0: 'obj/house_interior/house_interior.obj',
            1: 'obj/parking_lot/park.obj',
            2: 'obj/bathroom/bathroom.obj'
        }

        if self.selected_scene != -1:
            scene_path = scene_files[self.selected_scene]
            model.append(Obj(self.texture_shader, scene_path))

        # Create Object model
        obj_files = {
            0: 'obj/WusonOBJ.obj',
            1: 'obj/Porsche_911_GT2.obj',
            2: 'obj/bowl/model.obj'
        }

        if self.selected_obj != -1:
            obj_path = obj_files[self.selected_obj]
            model.append(Obj(self.phongex_shader, obj_path))

        self.add(model)

def main():
    viewer = Viewer()
    viewer.run()

if __name__ == '__main__':
    glfw.init()
    main()
    glfw.terminate()