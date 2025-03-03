import os.path

import numpy as np
from PyQt6.QtWidgets import QApplication, QFileDialog
import imgui
from pathlib import Path
import random
from imgui.integrations.glfw import GlfwRenderer
import math

from libs.buffer import *
from libs.camera import *
from libs.shader import *
from libs.transform import *

# from .object3D import *
# from .scene3D import *
from .object3D_v2 import *
from .scene3D_v2 import *
from .vcamera import *
from .sphere import *
from colormap import *
from utils import *


class Viewer:
    ''' Initialize attributes '''
    def __init__(self, args, img_width=640, img_height=480):
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

        self.scene_width, self.object_width, self.operation_width = 300, 300, 300
        self.scene_height = 160
        self.object_height = 160
        self.operation_height = 160

        self.camera_config_width, self.light_config_width, self.depth_config_width = 300, 300, 300
        self.camera_config_height = 160
        self.light_config_height = 160
        self.depth_config_height = 160

        self.rgb_view_width, self.depth_view_width = img_width, img_width
        self.rgb_view_height, self.depth_view_height = img_height, img_height

        self.win_width = self.scene_width + self.rgb_view_width*2 + self.camera_config_width
        self.win_height = self.rgb_view_height

        self.win = glfw.create_window(self.win_width, self.win_height, 'Viewer', None, None)
        if not self.win:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")

        glfw.make_context_current(self.win)

        # Initialize imgui
        imgui.create_context()
        self.imgui_impl = GlfwRenderer(self.win)
        # self.init_ui()
        self.font_size = 10
        self.load_font_size()

        # Enable depth testing
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDepthFunc(GL.GL_LESS)

        # bleding for RGBA
        GL.glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        # Initialize shaders
        self.phong_shader = Shader("shader/phong.vert", "shader/phong.frag")
        self.depth_texture_shader = Shader('shader/depth_texture.vert', 'shader/depth_texture.frag')
        self.object_shader = Shader('shader/depth_texture.vert', 'shader/depth_texture.frag')

        # Initial variable that manage objects
        self.drawables = []
        self.obj_management = {}
        
        # Initialize yaml file attributes
        self.yaw_range = []
        self.pitch_range = []
        self.center_translation_vec = glm.vec3(0.0, 0.0, 0.0)
        self.x_range = []
        self.y_range = []
        self.z_range = []
        self.scale_range = []
        self.scale_unit = 0 # defaut is 0 milimeters

        # Initialize mouse parameters
        # self.last_x = width / 2
        # self.last_y = height / 2
        self.first_mouse = True
        self.left_mouse_pressed = False
        self.right_mouse_pressed = False
        self.yaw = 0.0
        self.pitch = 0.0
        self.fov = 45.0

        # Initialize scene config
        self.selected_scene_path = args.scene_path
        self.selected_scene = None
        self.scene_view_flag = False
        self.bg_colors = [0.0, 0.0, 0.0]
        self.bg_changed = False
        self.warning_selected_path = True
        self.old_view = (0.0, 0.0)
        self.new_view = (5.0, 0.0)

        # Initialize object config
        self.rotate_obj_flag = False
        self.selected_obj_path = "No file selected"
        self.last_selected_object = None
        self.selected_object = None
        self.scale_changed = False
        self.scale_factor = 1
        self.prev_scale_factor = 1
        self.show_autoselect_window = False
        self.num_objs = args.num_objects
        self.obj_folder_path = args.obj_folder_path

        # Initialize operation
        self.vcameras = []
        self.move_camera_flag = False
        self.multi_cam_flag = False
        # self.time_save = 0.0
        # self.time_count = 0.0
        self.rgb_save_path = args.rgb_save_path
        self.depth_save_path = args.depth_save_path

        self.show_time_selection = False
        self.autosave_flag = False
        self.layout_opts = args.num_images
        self.save_all_flag = False
        self.show_autosave_window = False
        self.current_time = 0.0
        self.last_update_time = 0.0
        self.drag_object_flag = False
        self.translation = glm.vec3(0.0, 0.0, 0.0)
        self.reverse_translation = glm.vec3(0.0, 0.0, 0.0)
        self.obj_rotation_option = args.obj_rotation
        self.obj_location_option = args.obj_location
        self.obj_scale_option = args.obj_scale
        self.obj_texture_option = args.obj_texture
        self.camera_pos_option = args.camera_pos
        self.segmentation_flag = False

        # Initialize camera config
        self.cam_sys = None
        self.move_cam_sys_flag = False
        self.sphere_radius = 0.1
        self.cameraSpeed = 1
        self.old_cameraPos = glm.vec3(0.0, 0.0, 10.0)
        self.cameraPos_lst = []
        self.cameraPos = glm.vec3(0.0, 0.0, 10.0)
        self.cameraFront = glm.vec3(0.0, 0.0, 1.0)
        self.cameraUp = glm.vec3(0.0, 1.0, 0.0)
        self.lastFrame = 0.0
        self.selected_camera = None
        self.num_vcameras_changed = False
        self.num_vcameras = 10

        # Initialize light config
        self.shininess = 100
        self.lightPos = glm.vec3(250, 250, 300)
        self.lightColor = glm.vec3(1.0, 1.0, 1.0)
        self.lightPos_changed = False
        self.lightColor_changed = False

        # Initialize depth config
        self.near = 0.1
        self.far = 1000
        self.near_colors = [0.0, 0.0, 0.0]
        self.far_colors = [1.0, 1.0, 1.0]
        self.selected_colormap = 0
        
        # Initialize trackball
        self.trackball = Trackball()
        self.mouse = (0, 0)

        # Camera Intrinsic when convert from FOV
        # aspect_ratio = self.rgb_view_width / self.rgb_view_height
        # self.f_x = 640/(2*math.tan(glm.radians(45/2))*aspect_ratio)
        # self.f_y = 480/(2*math.tan(glm.radians(45/2)))
        # self.c_x = 640/2 + 300
        # self.c_y = 480/2
        # NYU rgb camera intrinsic https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html
        self.f_x = 518.86
        self.f_y = 519.47
        self.c_x = 325.58 + 300 # offset of viewport
        self.c_y = 253.74
        self.s = 0

        # using scenet layout
        self.sceneNet_usage = args.scene_net_layout
        print("self.sceneNet_usage", self.sceneNet_usage)

        # Auto mode, no need to click confirm to load anything
        if args.auto:
            self.process_scene_config()
            self.load_scene(self.sceneNet_usage)
            self.autoselect_obj()
            self.autosave(self.layout_opts)

        # Register callbacks
        glfw.set_key_callback(self.win, self.on_key)
        glfw.set_cursor_pos_callback(self.win, self.on_mouse_move)
        glfw.set_mouse_button_callback(self.win, self.mouse_button_callback)
        glfw.set_scroll_callback(self.win, self.scroll_callback)

        print('OpenGL', GL.glGetString(GL.GL_VERSION).decode() + ', GLSL',
              GL.glGetString(GL.GL_SHADING_LANGUAGE_VERSION).decode() +
              ', Renderer', GL.glGetString(GL.GL_RENDERER).decode())

        GL.glClearColor(1.0, 1.0, 1.0, 1.0)


    ''' Supportive functions '''
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
                delta_z = self.cameraSpeed * self.cameraFront

                if self.move_cam_sys_flag:
                    self.cam_sys.update_model_matrix(glm.translate(glm.mat4(1.0), -delta_z))

                if self.drag_object_flag:
                    translation_matrix = glm.translate(glm.mat4(1.0), -delta_z)
                    if self.selected_object:
                        current_model = self.selected_object.get_model_matrix()
                        model = current_model * translation_matrix
                        self.selected_object.update_attribute("model_matrix", model)
                else:
                    self.old_cameraPos = glm.vec3(self.cameraPos)
                    self.cameraPos += delta_z
                    self.trackball.update_cameraPos(-self.cameraPos, -self.old_cameraPos)

            if key == glfw.KEY_S:
                delta_z = self.cameraSpeed * self.cameraFront
                
                if self.move_cam_sys_flag:
                    self.cam_sys.update_model_matrix(glm.translate(glm.mat4(1.0), delta_z))

                if self.drag_object_flag:
                    translation_matrix = glm.translate(glm.mat4(1.0), delta_z)
                    if self.selected_object:
                        current_model = self.selected_object.get_model_matrix()
                        model = current_model * translation_matrix
                        self.selected_object.update_attribute("model_matrix", model)
                else:
                    self.old_cameraPos = glm.vec3(self.cameraPos)
                    self.cameraPos -= delta_z
                    self.trackball.update_cameraPos(-self.cameraPos, -self.old_cameraPos)

            if key == glfw.KEY_A:
                delta_x = glm.normalize(glm.cross(self.cameraFront, self.cameraUp)) * self.cameraSpeed
                if self.move_cam_sys_flag:
                    self.cam_sys.update_model_matrix(glm.translate(glm.mat4(1.0), -delta_x))
                
                self.old_cameraPos = glm.vec3(self.cameraPos)
                self.cameraPos -= delta_x
                self.trackball.update_cameraPos(-self.cameraPos, -self.old_cameraPos)

            if key == glfw.KEY_D:
                delta_x = glm.normalize(glm.cross(self.cameraFront, self.cameraUp)) * self.cameraSpeed
                if self.move_cam_sys_flag:
                    self.cam_sys.update_model_matrix(glm.translate(glm.mat4(1.0), delta_x))
                
                self.old_cameraPos = glm.vec3(self.cameraPos)
                self.cameraPos += delta_x
                self.trackball.update_cameraPos(-self.cameraPos, -self.old_cameraPos)

            for drawable in self.drawables:
                if hasattr(drawable, 'key_handler'):
                    drawable.key_handler(key)

    def on_mouse_move(self, window, xpos, ypos):
        """ Rotate on left-click & drag, pan on right-click & drag """
        old = self.mouse
        self.mouse = (xpos, glfw.get_window_size(window)[1] - ypos)

        if self.rotate_obj_flag and glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT):
            old, new = ((2 * np.asarray(pos) - glfw.get_window_size(window)) / glfw.get_window_size(window) for pos in (old, self.mouse))

            # Calculate delta movement
            delta_x = new[0] - old[0]
            delta_y = old[1] - new[1]
            
            # Update angles based on mouse movement
            sensitivity = 1  
            self.yaw = delta_x * sensitivity
            self.pitch = delta_y * sensitivity
            
            # Limit pitch to avoid gimbal lock
            # self.pitch = min(89.0, max(-89.0, self.pitch))

            current_model = self.selected_object.get_model_matrix()

            x_rotation_matrix = glm.rotate(glm.mat4(1.0), self.pitch, glm.vec3(1.0, 0.0, 0.0))
            y_rotation_matrix = glm.rotate(glm.mat4(1.0), self.yaw, glm.vec3(0.0, 1.0, 0.0))
            
            model = current_model * x_rotation_matrix * y_rotation_matrix

            # Uncomment this to rotate object but not successfully
            if self.selected_object:
                self.selected_object.update_attribute('model_matrix', model)

        elif self.drag_object_flag or self.move_cam_sys_flag:
            # Make sure that the left mouse is pressed when dragging the object
            if glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT):
                delta_x = self.mouse[0] - old[0]  # Calculate the horizontal movement delta
                delta_y = self.mouse[1] - old[1]  # Calculate the vertical movement delta

                # Create a translation matrix based on the mouse movement
                scale_factor = self.obj_management[self.selected_object.name]['scale_factor']
                scale_normalized_delta_x = delta_x / scale_factor
                scale_normalized_delta_y = delta_y / scale_factor
                translation_matrix = glm.translate(glm.mat4(1.0), glm.vec3(scale_normalized_delta_x, scale_normalized_delta_y, 0))

                current_model = self.selected_object.get_model_matrix()
                model = current_model * translation_matrix

                # Apply the translation to the selected object's model matrix
                if self.drag_object_flag and self.selected_object:
                    self.selected_object.update_attribute("model_matrix", model)
                if self.move_cam_sys_flag:
                    self.cam_sys.update_model_matrix(model)

        else:
            if glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT):
                # self.trackball.drag(old, self.mouse, glfw.get_window_size(window))
                self.trackball.drag(old, self.mouse, (self.depth_view_width, self.depth_view_height))

            if glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT):
                self.trackball.pan(old, self.mouse)

    def scroll_callback(self, window, xoffset, yoffset):
        # self.fov -= float(yoffset)
        # if self.fov < 1.0:
        #     self.fov = 1.0
        # if self.fov > 45.0:
        #     self.fov = 45.0
        self.trackball.zoom(yoffset, glfw.get_window_size(window)[1])

    def mouse_button_callback(self, window, button, action, mods):
        if button == glfw.MOUSE_BUTTON_LEFT:
            if action == glfw.PRESS:
                self.left_mouse_pressed = True
            elif action == glfw.RELEASE:
                self.left_mouse_pressed = False
        elif button == glfw.MOUSE_BUTTON_RIGHT:
            if action == glfw.PRESS:
                self.right_mouse_pressed = True
            elif action == glfw.RELEASE:
                self.right_mouse_pressed = False

    # def check_selected_object(self, cur_pos):
    #     ''' Function to check which object user want to drag '''
    #     for i,drawable in enumerate(self.drawables):
    #         if isinstance(drawable, Object):
    #             vertices = drawable.get_transformed_vertices() # list of vertices with type glm.vec3

    #             x_list = [vertex.x for vertex in vertices]
    #             min_x = min(x_list) # Smallest x value in vertices
    #             max_x = max(x_list) # Largest x value in vertices

    #             y_list = [vertex.y for vertex in vertices]
    #             np.savetxt('y.txt', y_list)
    #             min_y = min(y_list) # Smallest y value in vertices
    #             max_y = max(y_list) # Largest y value in vertices

    #             if (min_x <= cur_pos[0] and cur_pos[0] <= max_x and min_y <= cur_pos[1] and cur_pos[1] <= max_y):
    #                 return drawable

    def select_file(self, starting_folder):
        app = QApplication(sys.argv)
        file_path = QFileDialog.getOpenFileName(
            None,  # Parent widget (None for no parent)
            "Select File",  # Dialog title
            starting_folder  # Starting folder
        )[0]
        return file_path

    def select_folder(self):
        app = QApplication(sys.argv)
        folder_path = QFileDialog.getExistingDirectory()
        return folder_path
    
    def pass_magma_data(self, shader):
        # Flatten the array into 1D
        array_1d = magma_data.flatten()

        GL.glUseProgram(shader.render_idx)
        location = GL.glGetUniformLocation(shader.render_idx, "magma_data")

        # Pass the 1D array to the shader
        GL.glUniform1fv(location, len(array_1d), array_1d)

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

    # def framebuffer_segmentation(self):
    #     # Create Framebuffer for Segmentation Rendering
    #     segmentation_fbo = GL.glGenFramebuffers(1)
    #     segmentation_texture = GL.glGenTextures(1)
    #     GL.glBindTexture(GL.GL_TEXTURE_2D, segmentation_texture)
    #     GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGB, 800, 600, 0, GL.GL_RGB, GL.GL_UNSIGNED_BYTE, None)
    #     GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
    #     GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)

    #     # Attach Texture to FBO
    #     GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, segmentation_fbo)
    #     GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, segmentation_texture, 0)
    #     GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)  # Unbind
    #     # Random RGB color for each object
    #     color = random.

    def button_with_icon(self, image_path, button_text, icon_size=(20, 20)):
        """Create a button with an icon and text"""
        icon_texture = self.load_texture(image_path)

        # Calculate total button width (align + icon + spacing + text + align)
        text_size = imgui.calc_text_size(button_text)
        button_width = 2 + icon_size[0] + 8 + text_size.x + 2 # 8 pixels spacing and 2 for alignment
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

    def lay_object(self):
        if not self.selected_object or not self.selected_scene:
            return glm.mat4(1.0)
        object_bottom = self.selected_object.min_y
        scene_bottom = self.selected_scene.min_y
        print(f'selected scene min y: {scene_bottom}')
        print(f'selected object min y: {object_bottom}')

        if object_bottom > scene_bottom and scene_bottom < 0:
            translation_matrix = translate(0, (scene_bottom - object_bottom), 0)
        if object_bottom < scene_bottom and scene_bottom < 0:
            translation_matrix = translate(0, -(scene_bottom - object_bottom), 0)
        if object_bottom > scene_bottom and scene_bottom > 0:
            translation_matrix = translate(0, -(scene_bottom - object_bottom), 0)
        if object_bottom < scene_bottom and scene_bottom > 0:
            translation_matrix = translate(0, (scene_bottom - object_bottom), 0)

        return translation_matrix

    def load_font_size(self):
        io = imgui.get_io()
        io.font_global_scale = 1

    ''' Specialized Functions '''
    def save_rgb(self, save_path, numb):
        win_pos_width = self.scene_width
        pixels = GL.glReadPixels(win_pos_width, 0, self.rgb_view_width, self.rgb_view_height, GL.GL_RGB, GL.GL_UNSIGNED_BYTE)
        rgb_image = np.frombuffer(pixels, dtype=np.uint8).reshape((int(self.rgb_view_height), int(self.rgb_view_width), 3))

        # flip image up-down and left-right
        rgb_image = np.flipud(rgb_image)
        rgb_image = np.fliplr(rgb_image)

        # Convert numpy array (or your image data format) to PIL Image
        rgb_image = Image.fromarray(rgb_image)
        file_name = f"rgb_image_{numb}.png"
        rgb_image.save(os.path.join(save_path, file_name))
        print(f"Saved rgb image as {file_name}")

    def save_depth(self, save_path, numb):
        # Create a numpy array to hold the pixel data
        # win_pos_width = self.scene_width + self.rgb_view_width
        win_pos_width = self.scene_width

        ### Extract depth value ###
        # Read Pixel using GL_RGB
        # pixels = GL.glReadPixels(win_pos_width, 0, self.depth_view_width, self.depth_view_height, GL.GL_RGB, GL.GL_SHORT) # return linear depth, not raw depth value
        # depth_info = np.frombuffer(pixels, dtype=np.short).reshape((self.depth_view_height, self.depth_view_width, 3))
        #
        # # Flip the image vertically (because OpenGL's origin is at the bottom-left corner)
        # depth_info = np.flipud(depth_info)
        #
        # # Get metric depth value for image
        # depth_info = depth_info[:,:,0] # get only 1 channel (gray image)
        # np.savetxt('depth.txt',depth_info)

        ### Save depth image ###
        # depth_pixels = GL.glReadPixels(
        #         win_pos_width, 0,
        #         self.rgb_view_width, self.rgb_view_height,
        #         GL.GL_RGB, GL.GL_UNSIGNED_BYTE
        #     )

        # depth_image = np.frombuffer(depth_pixels, dtype=np.uint8).reshape(
        #     (int(self.rgb_view_height), int(self.rgb_view_width), 3)
        # )
        #
        # # Flip the image vertically (because OpenGL's origin is at the bottom-left corner)
        # depth_image = np.flipud(depth_image)
        # depth_image_pil = Image.fromarray(depth_image[:,:,0], mode='L')

        depth_pixels = GL.glReadPixels(win_pos_width, 0,
            self.rgb_view_width, self.rgb_view_height,
            GL.GL_DEPTH_COMPONENT,
            GL.GL_FLOAT
        )
        depth_array = np.frombuffer(depth_pixels, dtype=np.float32).reshape(self.rgb_view_height, self.rgb_view_width)
        background_mask = (depth_array == 1) # fragment which are not drawn

        px, py = np.meshgrid(np.arange(self.rgb_view_width), np.arange(self.rgb_view_height)) # meshgrih to calculate x,y offset pixel
        x_ndc = (px / self.rgb_view_width) * 2 - 1
        y_ndc = (py / self.rgb_view_height) * 2 - 1

        z_ndc = depth_array * 2.0 - 1.0 # invert viewport (to NDC -1 -> 1)
        real_depth = ((2.0 * self.near * self.far) / (self.far + self.near - z_ndc * (self.far - self.near))) # invert projection

        x_camera = x_ndc * real_depth / self.f_x
        y_camera = y_ndc * real_depth / self.f_y

        euclidean_depth = np.sqrt(x_camera ** 2 + y_camera ** 2 + real_depth ** 2)

        euclidean_depth[background_mask] = 0

        # flip due to CV camera
        depth_image = np.flipud(euclidean_depth)
        depth_image = np.fliplr(depth_image)

        depth_image = depth_image*self.scale_unit # convert to milimeter
        depth_image = depth_image.astype(np.uint16)

        # check if the depth valid
        if not valid_depth_map(depth_image):
            print("Invalid depth map")
            return False


        depth_image_pil = Image.fromarray(depth_image, mode='I;16')

        # Create a unique file name using timestamp
        file_name = f"depth_image_{numb}.png"

        # Save the image to the selected directory
        depth_image_pil.save(os.path.join(save_path, file_name))
        print(f"Saved depth image as {file_name}")
        return True

    def autosave(self, lay_opts):
        scene_name = os.path.basename(os.path.dirname(self.selected_scene_path))
        if self.rgb_save_path == '':
            self.rgb_save_path = Path("./dataset/rgb") / scene_name
            self.rgb_save_path.mkdir(parents=True, exist_ok=True)

        if self.depth_save_path == '':
            self.depth_save_path = Path("./dataset/depth") / scene_name
            self.depth_save_path.mkdir(parents=True, exist_ok=True)

        i = 0
        while i < lay_opts:
            if self.camera_pos_option:
                self.cameraPos.x = random.uniform(self.x_range[0], self.x_range[1])
                self.cameraPos.y = random.uniform(self.y_range[0], self.y_range[1])
                self.cameraPos.z = random.uniform(self.z_range[0], self.z_range[1])

            lookAt_position = glm.vec3(random.uniform(self.x_range[0], self.x_range[1]),
                                       random.uniform(self.y_range[0], self.y_range[1]),
                                       random.uniform(self.z_range[0], self.z_range[1])
                                       )
            # lookAt_position = glm.vec3(0, 0, 100)

            view = glm.lookAt(self.cameraPos, lookAt_position, glm.vec3(0, -1, 0)) # flip due to cv camera

            # yaw = random.choice(np.linspace(self.yaw_range[1], self.yaw_range[0], 20))
            # pitch = random.choice(np.linspace(self.pitch_range[1], self.pitch_range[0], 20))
            # direction = glm.vec3(
            #     glm.cos(glm.radians(yaw)) * glm.cos(glm.radians(pitch)),  # x
            #     glm.sin(glm.radians(pitch)),  # y
            #     glm.sin(glm.radians(yaw)) * glm.cos(glm.radians(pitch))  # z
            # )

            # self.cameraFront = glm.normalize(direction)
            # view = glm.lookAt(self.cameraPos, self.cameraFront, glm.vec3(0, -1, 0))

            GL.glClearColor(0, 0, 0, 1.0)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

            # Viewport for RGB Scene
            rgb_win_pos_x = self.scene_width
            rgb_win_pos_y = self.win_height - self.rgb_view_height # start from bottom-left

            GL.glViewport(rgb_win_pos_x, rgb_win_pos_y, self.rgb_view_width, self.rgb_view_height)
            GL.glScissor(rgb_win_pos_x, rgb_win_pos_y, self.rgb_view_width, self.rgb_view_height)
            GL.glEnable(GL.GL_SCISSOR_TEST)
            GL.glClearColor(*self.bg_colors, 1.0)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT)
            GL.glUseProgram(self.depth_texture_shader.render_idx)

            projection_mat = self.cam_proj_matrix(
                rgb_win_pos_x,
                rgb_win_pos_y,
                self.rgb_view_width,
                self.rgb_view_height,
                self.near,
                self.far
            )        # depth viewport also use win pos of rgb to ensure the two show the same
            # projection_mat = glm.perspective(glm.radians(45), (self.rgb_view_width/ self.rgb_view_height), self.near, self.far)

            for drawable in self.drawables:
                if self.obj_location_option and isinstance(drawable, Object):
                    # Instead of creating a translation matrix and multiply with current model, 
                    # I add the translation vector directly to the last column of current model 
                    # (avoid effect of scaling factor and rotation to the translation vector)
                    x_obj = random.uniform(self.x_range[0], self.x_range[1])
                    y_obj = random.uniform(self.y_range[0], self.y_range[1])
                    z_obj = random.uniform(self.z_range[0], self.z_range[1])
                    
                    self.reverse_translation = self.obj_management[drawable.name]['reverse_translation']
                    self.translation = glm.vec3(x_obj,y_obj,z_obj)
                    
                    current_model = drawable.get_model_matrix()
                    current_model[3] = current_model[3] + glm.vec4(self.reverse_translation, 0.0) + glm.vec4(self.translation, 0.0)
                    
                    self.obj_management[drawable.name]['reverse_translation'] = - self.translation
                    
                    drawable.update_attribute('model_matrix', current_model)

                if self.obj_rotation_option and isinstance(drawable, Object):
                    yaw = random.choice(np.linspace(0,360, 360))
                    pitch = random.choice(np.linspace(0,360, 360))
                    
                    x_rotation_matrix = glm.rotate(glm.mat4(1.0), glm.radians(-pitch), glm.vec3(1.0, 0.0, 0.0)) # add - angle because object rotation not camera
                    y_rotation_matrix = glm.rotate(glm.mat4(1.0), glm.radians(-yaw), glm.vec3(0.0, 1.0, 0.0))
                    
                    current_model = drawable.get_model_matrix()
                    model = current_model * x_rotation_matrix * y_rotation_matrix

                    drawable.update_attribute('model_matrix', model)

                if self.obj_scale_option and isinstance(drawable, Object):
                    self.scale_factor = random.uniform(self.scale_range[0], self.scale_range[1])
                    self.prev_scale_factor = self.obj_management[drawable.name]['scale_factor']
                    self.obj_management[drawable.name]['scale_factor'] = self.scale_factor
                    # print(f'name: {drawable.name}')
                    # print(f'scale factor: {self.obj_management[drawable.name]["scale_factor"]}')

                    current_model = drawable.get_model_matrix()
                    prev_scale_matrix = glm.scale(glm.mat4(1.0), glm.vec3(1/self.prev_scale_factor))
                    scale_matrix = glm.scale(glm.mat4(1.0), glm.vec3(self.scale_factor))
                    model = current_model * prev_scale_matrix * scale_matrix # to scale back before applying new scale

                    drawable.update_attribute('model_matrix', model)

                # switch texture every 500
                if (self.obj_texture_option
                        and isinstance(drawable, Object)
                        and (i % 500 == 0)
                        and (i != 0)):
                    # Only update map_Kd, supposed an object only have 1 simple texture
                    texture_list = os.listdir('textures')
                    for j, texture in enumerate(texture_list):
                        texture_list[j] = os.path.join('textures', texture)
                    # Store the initial texture of the object
                    texture_list.append(drawable.get_texture())
                    path = random.choice(texture_list)
                    drawable.update_attribute('texture', path)

                drawable.set_mode(1) # mode for rgb image

                # update light configuration
                if self.lightPos_changed:
                    drawable.update_lightPos(self.lightPos)

                if self.lightColor_changed:
                    drawable.update_lightColor(self.lightColor)

                drawable.update_shininess(self.shininess)

                drawable.update_attribute('view_matrix', view)
                # projection = glm.perspective(glm.radians(self.fov), self.rgb_view_width / self.rgb_view_height, self.near, self.far)
                drawable.update_attribute('projection_matrix', projection_mat)

                # Normal rendering
                drawable.draw(self.cameraPos)

            # # Viewport for Depth Scene
            # depth_win_pos_x = self.scene_width + self.rgb_view_width
            # depth_win_pos_y = self.win_height - self.depth_view_height # start from bottom-left
            # GL.glViewport(depth_win_pos_x, depth_win_pos_y, self.depth_view_width, self.depth_view_height)
            # GL.glScissor(depth_win_pos_x, depth_win_pos_y, self.depth_view_width, self.depth_view_height)
            # GL.glClear(GL.GL_COLOR_BUFFER_BIT)
            #
            # for drawable in self.drawables:
            #     drawable.set_mode(0) # mode for depth map
            #
            #     # update depth map color
            #     drawable.update_colormap(self.selected_colormap)
            #
            #     # Depth map rendering
            #     drawable.update_near_far(self.near, self.far)
            #
            #     # Draw the full object
            #     drawable.draw(self.cameraPos)
            #
            #     # Visualize with chosen colormap
            #     if self.selected_colormap == 1:
            #         self.pass_magma_data(self.depth_texture_shader)

            GL.glDisable(GL.GL_SCISSOR_TEST)

            status = self.save_depth(self.depth_save_path, i)
            if not status:
                glfw.swap_buffers(self.win)
                continue

            self.save_rgb(self.rgb_save_path, i)
            i += 1

            glfw.swap_buffers(self.win)
        self.autosave_flag = False

    def autoselect_obj(self):
        for i in range(self.num_objs):
            # List all subdirectories (object categories)
            subdirs = [d for d in os.listdir(self.obj_folder_path) if os.path.isdir(os.path.join(self.obj_folder_path, d))]
            
            if not subdirs:
                print("No subdirectories found in", self.obj_folder_path)
                return None

            # Randomly select a subdirectory
            random_subdir = random.choice(subdirs)
            subdir_path = os.path.join(self.obj_folder_path, random_subdir)

            # List all .obj files in the selected subdirectory
            obj_files = [f for f in os.listdir(subdir_path) if f.endswith('.obj')]

            if not obj_files:
                print(f"No .obj files found in {subdir_path}")
                return None

            # Randomly select an .obj file
            random_obj_file = random.choice(obj_files)
            file_path = os.path.join(subdir_path, random_obj_file)

            self.load_object(file_path)

    '''
    This function is used to debug
    '''
    def random_location(self):
        # x_min, y_min, z_min = self.selected_scene.min_x, self.selected_scene.min_y, self.selected_scene.min_z
        # x_max, y_max, z_max = self.selected_scene.max_x, self.selected_scene.max_y, self.selected_scene.max_z

        x_min, x_max = self.x_range[0], self.x_range[1]
        y_min, y_max = self.y_range[0], self.y_range[1]
        z_min, z_max = self.z_range[0], self.z_range[1]

        print(f'x min: {x_min}, y min: {y_min}, z min: {z_min}')
        print(f'x max: {x_max}, y max: {y_max}, z max: {z_max}')
        GL.glClearColor(0, 0, 0, 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        # Viewport for RGB Scene
        win_pos_width = self.scene_width
        win_pos_height = self.win_height - self.rgb_view_height # start from bottom-left

        GL.glViewport(win_pos_width, win_pos_height, self.rgb_view_width, self.rgb_view_height)
        GL.glScissor(win_pos_width, win_pos_height, self.rgb_view_width, self.rgb_view_height)
        GL.glEnable(GL.GL_SCISSOR_TEST)
        GL.glClearColor(*self.bg_colors, 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)
        GL.glUseProgram(self.depth_texture_shader.render_idx)

        for drawable in self.drawables:
            drawable.set_mode(1) # mode for rgb image

            # update light configuration
            if self.lightPos_changed:
                drawable.update_lightPos(self.lightPos)

            if self.lightColor_changed:
                drawable.update_lightColor(self.lightColor)

            drawable.update_shininess(self.shininess)

            if isinstance(drawable, Object):
                x = random.uniform(x_min, x_max)
                z = random.uniform(z_min, z_max)

                current_model = drawable.get_model_matrix()

                # Instead of creating a translation matrix and multiply with current model, 
                # I add the translation vector directly to the last column of current model 
                # (avoid effect of scaling factor and rotation to the translation vector)
                self.reverse_translation = self.obj_management[drawable.name]['reverse_translation']
                self.translation = glm.vec3(x,0,z)
                current_model[3] = current_model[3] + glm.vec4(self.reverse_translation, 0.0) + glm.vec4(self.translation, 0.0)
                self.obj_management[drawable.name]['reverse_translation'] = - self.translation

                yaw = random.choice(np.linspace(0, 360, 360))
                pitch = random.choice(np.linspace(0, 360, 360))
                x_rotation_matrix = glm.rotate(glm.mat4(1.0), glm.radians(-pitch), glm.vec3(1.0, 0.0, 0.0)) # add - angle because object rotation not camera
                y_rotation_matrix = glm.rotate(glm.mat4(1.0), glm.radians(-yaw), glm.vec3(0.0, 1.0, 0.0))

                model = current_model * x_rotation_matrix * y_rotation_matrix

                drawable.update_attribute('model_matrix', model)

            # drawable.update_attribute('view_matrix', view)

            projection = glm.perspective(glm.radians(self.fov), self.rgb_view_width / self.rgb_view_height, self.near, self.far)
            drawable.update_attribute('projection_matrix', projection)

            # Normal rendering
            drawable.draw(self.cameraPos)

        # Viewport for Depth Scene
        win_pos_width = self.scene_width + self.rgb_view_width
        win_pos_height = self.win_height - self.depth_view_height # start from bottom-left
        GL.glViewport(win_pos_width, win_pos_height, self.depth_view_width, self.depth_view_height)
        GL.glScissor(win_pos_width, win_pos_height, self.depth_view_width, self.depth_view_height)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)

        for drawable in self.drawables:
            drawable.set_mode(0) # mode for depth map

            # update depth map color
            drawable.update_colormap(self.selected_colormap)

            # drawable.update_attribute('view_matrix', view)

            projection = glm.perspective(glm.radians(self.fov), self.depth_view_width / self.depth_view_height, self.near, self.far)
            drawable.update_attribute('projection_matrix', projection)

            # Depth map rendering
            drawable.update_near_far(self.near, self.far)

            # Draw the full object
            drawable.draw(self.cameraPos)

            # Visualize with chosen colormap
            if self.selected_colormap == 1:
                self.pass_magma_data(self.depth_texture_shader)

        GL.glDisable(GL.GL_SCISSOR_TEST)

        glfw.swap_buffers(self.win)

    def multi_cam(self):
        # Define the hemisphere of multi-camera
        self.cam_sys = Sphere(self.phong_shader).setup()
        self.cam_sys.radius = self.sphere_radius
        self.cam_sys.generate_sphere() # update vertices with new radius

        self.vcameras = []
        golden_ratio = (1 + np.sqrt(5)) / 2
        for i in range(self.num_vcameras):
            vcamera = VCamera(self.phong_shader).setup()
            vcamera.update_label(i)

            # Generate points using Fibonacci sphere algorithm
            y = 1 - (i / float(self.num_vcameras - 1)) # Only use top half (y >= 0)
            y = max(0.001, y)  # Ensure we stay in top hemisphere
            
            radius_at_y = np.sqrt(1 - y * y)
            theta = 2 * np.pi * i / golden_ratio
            
            # Convert to Cartesian coordinates
            x = self.cam_sys.radius * radius_at_y * np.cos(theta)
            z = self.cam_sys.radius * radius_at_y * np.sin(theta)
            y = self.cam_sys.radius * y
            
            P = glm.vec3(x, y, z)

            # Set up view matrix for camera
            eye = P
            at = P + glm.normalize(P) # Point in the direction of the outward ray
            up = glm.normalize(glm.vec3(vcamera.model[1]))

            vcamera.view = glm.lookAt(eye, at, up)
            vcamera.projection = glm.perspective(glm.radians(self.fov), self.rgb_view_width / self.rgb_view_height, 0.1, 100.0)

            self.cameraPos_lst.append(eye) # to update cameraPos
            self.vcameras.append(vcamera) # list contains all virtual cameras

    def multi_cam_inward(self):
        # Define the hemisphere of multi-camera
        self.cam_sys = Sphere(self.phong_shader).setup()
        self.cam_sys.radius = self.sphere_radius
        self.cam_sys.generate_sphere() # update vertices with new radius

        self.vcameras = []
        golden_ratio = (1 + np.sqrt(5)) / 2
        for i in range(self.num_vcameras):
            vcamera = VCamera(self.phong_shader).setup()
            vcamera.update_label(i)

            # Generate points using Fibonacci sphere algorithm
            y = 1 - (i / float(self.num_vcameras - 1)) # Only use top half (y >= 0)
            y = max(0.001, y)  # Ensure we stay in top hemisphere
            
            radius_at_y = np.sqrt(1 - y * y)
            theta = 2 * np.pi * i / golden_ratio
            
            # Convert to Cartesian coordinates
            x = self.cam_sys.radius * radius_at_y * np.cos(theta)
            z = self.cam_sys.radius * radius_at_y * np.sin(theta)
            y = self.cam_sys.radius * y
            
            P = glm.vec3(x, y, z)

            # Set up view matrix for camera
            eye = P
            at = glm.vec3(0,0,0) # Point in the direction of the outward ray
            up = glm.normalize(glm.vec3(vcamera.model[1]))

            vcamera.view = glm.lookAt(eye, at, up)
            vcamera.projection = glm.perspective(glm.radians(self.fov), self.rgb_view_width / self.rgb_view_height, 0.1, 100.0)

            self.cameraPos_lst.append(eye) # to update cameraPos
            self.vcameras.append(vcamera) # list contains all virtual cameras

    def process_scene_config(self):
        dir_path = './config/'
        file_path = dir_path + os.path.basename(self.selected_scene_path)[:-3] + 'yaml'
        if os.path.exists(file_path):
            content = read_yaml_file(file_path)
            for key, value in content.items():
                if key == "cameraPos" and isinstance(value, list):  
                    setattr(self, key, glm.vec3(*value))  # Convert list to glm.vec3
                    self.trackball.update_cameraPos(self.cameraPos) # Update trackball cameraPos
                elif key == "center_translation_vec" and isinstance(value, list): 
                    setattr(self, key, glm.vec3(*value)) # Convert list to glm.vec3
                elif hasattr(self, key):  
                    setattr(self, key, value)

        else:
            print(f'{file_path} is not found. Use default!')

    def reset(self):
        self.obj_location_option = True
        # Remove all scenes and objects
        self.drawables.clear()
        self.autosave_flag = False
        self.multi_cam_flag = False
        self.move_camera_flag = False
        self.drag_object_flag = False
        self.move_cam_sys_flag = False

    def load_scene(self, scene_net_flag):
        model = []

        # remove last Scene
        self.drawables = [drawable for drawable in self.drawables if not isinstance(drawable, Scene)]

        # Add chosen object or scene
        if self.selected_scene_path != "No file selected":
            self.selected_scene = Scene(self.depth_texture_shader, self.selected_scene_path, scene_net_flag)

            # Translate the scene for better view
            current_model = self.selected_scene.get_model_matrix()
            translation_matrix = glm.translate(glm.mat4(1.0), self.center_translation_vec)
            model_matrix = current_model * translation_matrix
            self.selected_scene.update_attribute("model_matrix", model_matrix)

            model.append(self.selected_scene)

        self.add(model)

    def load_object(self, file_path):
        model = []
        self.selected_object = Object(self.object_shader, file_path)

        # current_model = self.selected_object.get_model_matrix()
        # translation_matrix = glm.translate(glm.mat4(1.0), self.center_translation_vec)
        # model_matrix = current_model * translation_matrix
        # self.selected_scene.update_attribute("model_matrix", model_matrix)
        # translation_matrix = self.lay_object()
        # model_matrix = current_model * translation_matrix
        # self.selected_object.update_attribute('model_matrix', model_matrix)

        model.append(self.selected_object)

        # Add new object to object management
        init_name = self.selected_object.name + '_' + str(0)
        if init_name in self.obj_management:
            # Track order of object
            self.obj_management[init_name]['order'] += 1

            # Update new name
            self.selected_object.name = self.selected_object.name + '_' + str(self.obj_management[init_name]['order'])
            self.obj_management[self.selected_object.name] = {
                'scale_factor': 1,
                'reverse_translation': glm.vec3(0.0, 0.0, 0.0),
                'translation': glm.vec3(0.0, 0.0, 0.0)
            }
        else:
            # First object will store order to manage obj name
            self.selected_object.name = self.selected_object.name + '_' + str(0)
            self.obj_management[self.selected_object.name] = {
                'order': 0,
                'scale_factor': 1,
                'reverse_translation': glm.vec3(0.0, 0.0, 0.0),
                'translation': glm.vec3(0.0, 0.0, 0.0)
            }

        # Set scale factor for new object
        self.prev_scale_factor = 1
        self.scale_factor = 1

        self.add(model)

    def use_trackball(self):
        self.drag_object_flag = False # reset drag drop object
        for drawable in self.drawables:
            win_size = glfw.get_window_size(self.win)
            drawable.view = self.trackball.view_matrix()
            drawable.projection = self.trackball.projection_matrix(win_size)

    # def move_camera_around(self):
    #     '''
    #     Set the master camera on the sphere cover the scene. The radius of the sphere is detemined by max of x / z values + offset
    #     '''
    #     sphere = Sphere(self.phong_shader).setup()

    #     # for drawable in self.drawables:
    #     #     if isinstance(drawable, Scene):
    #     #         if drawable.max_z > drawable.max_x:
    #     #             sphere.radius = drawable.max_z + (1 / 3) * (drawable.max_z - drawable.min_z)
    #     #         else:
    #     #             sphere.radius = drawable.max_x + (1 / 3) * (drawable.max_x - drawable.min_x)
    #     #     break
        
    #     sphere.radius = self.sphere_radius
    #     sphere.generate_sphere() # update vertices with new radius
    #     return random.choice(sphere.vertices)

    def render(self):
        GL.glClearColor(0, 0, 0, 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        # Viewport for RGB Scene
        rgb_win_pos_x = self.scene_width
        rgb_win_pos_y = self.win_height - self.rgb_view_height # this is 0

        GL.glEnable(GL.GL_SCISSOR_TEST)
        GL.glViewport(rgb_win_pos_x, rgb_win_pos_y, self.rgb_view_width, self.rgb_view_height)
        GL.glScissor(rgb_win_pos_x, rgb_win_pos_y, self.rgb_view_width, self.rgb_view_height)

        GL.glClearColor(*self.bg_colors, 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)
        GL.glUseProgram(self.depth_texture_shader.render_idx)

        if self.multi_cam_flag:
            # self.multi_cam()
            self.multi_cam_inward()

        projection_mat = self.cam_proj_matrix(
            rgb_win_pos_x,
            rgb_win_pos_y,
            self.rgb_view_width,
            self.rgb_view_height,
            self.near,
            self.far
        )

        for drawable in self.drawables:
            drawable.set_mode(1) # mode for rgb image

            # update light configuration
            if self.lightPos_changed:
                drawable.update_lightPos(self.lightPos)

            if self.lightColor_changed:
                drawable.update_lightColor(self.lightColor)

            drawable.update_shininess(self.shininess)

            # Define model matrix
            if self.selected_object == drawable and self.scale_changed and self.scale_factor != 0:
                current_model = self.selected_object.get_model_matrix()
                
                prev_scale_matrix = glm.scale(glm.mat4(1.0), glm.vec3(1/self.prev_scale_factor))
                scale_matrix = glm.scale(glm.mat4(1.0), glm.vec3(self.scale_factor))

                model = current_model * prev_scale_matrix * scale_matrix # to scale back before applying new scale

                self.prev_scale_factor = self.scale_factor
                print(f'prev scale factor: {self.prev_scale_factor}')
                drawable.update_attribute('model_matrix', model)

            # Define view matrix
            view = self.trackball.view_matrix()
            # view = np.array(
            #     [
            #         [1., 0., 0., 0.],
            #         [0., 1., 0., 0.],
            #         [0., 0., 1., 0.],
            #         [0., 0., 0., 1.]
            #     ], dtype = np.float32
            # )

            if self.move_camera_flag:
                # Call to create hemisphere of multi-cam
                self.multi_cam()

                self.current_time = glfw.get_time()
                if self.current_time - self.last_update_time >= 0.5:  # Check if 1 second has passed
                    vcamera = random.choice(self.vcameras)
                    self.cameraPos = self.cameraPos_lst[self.vcameras.index(vcamera)]
                    view = vcamera.view
                    self.last_update_time = self.current_time

            if self.selected_camera:
                view = self.selected_camera.view

            # projection_mat = self.trackball.projection_matrix((self.rgb_view_width, self.rgb_view_height))
            drawable.update_attribute('view_matrix', view)
            drawable.update_attribute('projection_matrix', projection_mat)

            # print("View matrix \n", view)
            # print("Projection matrix \n", projection_mat)


            # Normal rendering
            drawable.draw(self.cameraPos)

        # Viewport for Depth Scene
        depth_win_pos_x = self.scene_width + self.rgb_view_width
        depth_win_pos_y = self.win_height - self.depth_view_height # start from bottom-left
        GL.glViewport(depth_win_pos_x, depth_win_pos_y, self.depth_view_width, self.depth_view_height)
        GL.glScissor(depth_win_pos_x, depth_win_pos_y, self.depth_view_width, self.depth_view_height)
        # GL.glClearColor(*self.bg_colors, 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)

        for drawable in self.drawables:
            drawable.set_mode(0) # mode for depth map

            # update depth map color
            drawable.update_colormap(self.selected_colormap)

            # Define model matrix
            if self.selected_object == drawable and self.scale_changed and self.scale_factor != 0:
                current_model = self.selected_object.get_model_matrix()
                
                prev_scale_matrix = glm.scale(glm.mat4(1.0), glm.vec3(1/self.prev_scale_factor))
                scale_matrix = glm.scale(glm.mat4(1.0), glm.vec3(self.scale_factor))

                model = current_model * prev_scale_matrix * scale_matrix # to scale back before applying new scale

                self.prev_scale_factor = self.scale_factor
                drawable.update_attribute('model_matrix', model)

            # Define view matrix
            view = self.trackball.view_matrix()

            if self.move_camera_flag:
                # Call to create hemisphere of multi-cam
                self.multi_cam()

                current_time = glfw.get_time()
                if current_time - self.last_update_time >= 0.5:  # Check if 1 second has passed
                    vcamera = random.choice(self.vcameras)
                    self.cameraPos = self.cameraPos_lst[self.vcameras.index(vcamera)]
                    view = vcamera.view
                    self.last_update_time = current_time

            if self.selected_camera:
                view = self.selected_camera.view
            drawable.update_attribute('view_matrix', view)

            # Depth map rendering
            drawable.update_near_far(self.near, self.far)

            # Draw the full object
            drawable.draw(self.cameraPos)

            # Visualize with chosen colormap
            if self.selected_colormap == 1:
                self.pass_magma_data(self.depth_texture_shader)

        GL.glDisable(GL.GL_SCISSOR_TEST)

    # def render_segmentation(self):
    #     if self.segmentation_flag:

    ''' User Interface '''
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

        select_scene_flag = self.button_with_icon('icons/load.png', 'Select Scene')
        if select_scene_flag:
            self.selected_scene_path = self.select_file('./scene')

        imgui.text(f"{self.selected_scene_path}")

        # if imgui.button('Scene View'):
        #     scene_name = self.selected_scene.name
        #     self.rgb_save_path = Path("./rgb") / scene_name
        #     self.depth_save_path = Path("./depth") / scene_name

        #     # Create directories if they don't exist
        #     self.rgb_save_path.mkdir(parents=True, exist_ok=True)
        #     self.depth_save_path.mkdir(parents=True, exist_ok=True)

        #     self.scene_view_flag = not self.scene_view_flag

        # Warning if not select
        if self.rgb_save_path == "" or self.depth_save_path == "":
            imgui.text('Please select save path')

        # Adjust RGB
        self.bg_changed, self.bg_colors = imgui.input_float3('BG Color', self.bg_colors[0], self.bg_colors[1], self.bg_colors[2], format='%.2f')

        font_size = imgui.get_font_size()
        vertical_padding = 8
        button_height = font_size + vertical_padding*2
        imgui.set_cursor_pos((imgui.get_window_width()//4, imgui.get_window_height() - button_height))
        imgui.set_next_item_width(imgui.get_window_width()//2)
        if imgui.button("Load Scene"):
            self.trackball.set_default()
            self.process_scene_config()
            self.load_scene(self.sceneNet_usage)

        imgui.end()

        ########################################################################
        #                                Object                                #
        ########################################################################
        
        win_pos_width = 0
        win_pos_height = self.scene_height

        if not self.show_autoselect_window:
            imgui.set_next_window_position(win_pos_width, win_pos_height)
            imgui.set_next_window_size(self.object_width, self.object_height)
            imgui.begin("Object")

            if imgui.button("AutoSelect"):
                self.show_autoselect_window = True

            select_obj_flag = self.button_with_icon('icons/load.png', 'Select Obj')
            if select_obj_flag:
                self.selected_obj_path = self.select_file('./object')
            imgui.text(f"{self.selected_obj_path}")

            imgui.set_next_item_width(100)

            current_item = "" if self.selected_object is None else self.selected_object.name
            previous_selected_object = self.selected_object  # Store the previous selection

            if imgui.begin_combo("List of Objects", current_item):
                for drawable in self.drawables:
                    if isinstance(drawable, Object):
                        is_selected = self.selected_object == drawable

                        # Check if a new object is selected
                        if imgui.selectable(drawable.name, is_selected)[0]:
                            if self.selected_object != drawable:  # Only update if it's a new selection
                                self.selected_object = drawable

                                # Reset scale factor only if a different object is selected
                                self.prev_scale_factor = self.obj_management[self.selected_object.name]['scale_factor']
                                self.scale_factor = self.obj_management[self.selected_object.name]['scale_factor']

                        if is_selected:
                            imgui.set_item_default_focus()

                imgui.end_combo()

            if self.selected_object:
                imgui.text(f"{self.selected_object.name} is selected")
            else:
                imgui.text(f"No object is selected")

            imgui.set_next_item_width(100)
            self.scale_changed, self.scale_factor = imgui.input_float("Scale factor", self.scale_factor, step=0.1, format="%.2f")
            if self.selected_object:
                self.obj_management[self.selected_object.name]['scale_factor'] = self.scale_factor # update info for obj

            if imgui.button('Random Textures'):
                texture_list = os.listdir('textures')
                for i, texture in enumerate(texture_list):
                    texture_list[i] = os.path.join('textures', texture)
                # Store the initial texture of the object
                texture_list.append(self.selected_object.get_texture())
                path = random.choice(texture_list)
                self.selected_object.update_attribute('texture', path)

            if self.drag_object_flag:
                imgui.push_style_color(imgui.COLOR_BUTTON, 0.6, 0.8, 0.6, 1.0)  # Green when clicked
            else:
                imgui.push_style_color(imgui.COLOR_BUTTON, 1.0, 0.0, 0.0, 1.0)  # Red before clicking

            if imgui.button("Drag/Drop"):
                self.drag_object_flag = not self.drag_object_flag # Toggle the flag

                hand_cursor = glfw.create_standard_cursor(glfw.HAND_CURSOR)
                arrow_cursor = glfw.create_standard_cursor(glfw.ARROW_CURSOR)

                if self.drag_object_flag:
                    # Set the hand cursor for the window
                    glfw.set_cursor(self.win, hand_cursor)
                else:
                    # Set the hand cursor for the window
                    glfw.set_cursor(self.win, arrow_cursor)

            # Restore default button color
            imgui.pop_style_color()

            if self.rotate_obj_flag:
                imgui.push_style_color(imgui.COLOR_BUTTON, 0.6, 0.8, 0.6, 1.0)  # Green when clicked
            else:
                imgui.push_style_color(imgui.COLOR_BUTTON, 1.0, 0.0, 0.0, 1.0)  # Red before clicking

            if imgui.button('Rotate Object'):
                self.rotate_obj_flag = not self.rotate_obj_flag # Toggle the flag

                hand_cursor = glfw.create_standard_cursor(glfw.HAND_CURSOR)
                arrow_cursor = glfw.create_standard_cursor(glfw.ARROW_CURSOR)

                if self.rotate_obj_flag:
                    # Set the hand cursor for the window
                    glfw.set_cursor(self.win, hand_cursor)
                else:
                    # Set the hand cursor for the window
                    glfw.set_cursor(self.win, arrow_cursor)

            # Restore default button color
            imgui.pop_style_color()

            # font_size = imgui.get_font_size()
            # vertical_padding = 8
            # button_height = font_size + vertical_padding*2
            # imgui.set_cursor_pos((imgui.get_window_width()//4, imgui.get_window_height() - button_height))
            # imgui.set_next_item_width(imgui.get_window_width()//2)
            if imgui.button("Load Object"):
                self.load_object(self.selected_obj_path)

            imgui.end()
        else:
            # ==========================
            # AUTOSELECT CONFIG WINDOW
            # ==========================
            imgui.set_next_window_position(win_pos_width, win_pos_height)
            imgui.set_next_window_size(self.object_width, self.object_height)
            imgui.begin("AutoSelect Configs")

            imgui.spacing()
            window_width = imgui.get_window_width()
            text_width = imgui.calc_text_size("AutoSelect Configs").x
            text_pos = (window_width - text_width) / 2
            imgui.set_cursor_pos_x(text_pos)
            imgui.text('AutoSelect Configs')

            imgui.separator()

            if imgui.button("Object Folder"):
                self.obj_folder_path = self.select_folder()

            imgui.set_next_item_width(100)
            num_objs_changed, self.num_objs = imgui.input_int("Number Of Objects", self.num_objs)

            if imgui.button("Confirm"):
                self.autoselect_obj()

            # ======= RETURN TO MAIN WINDOW BUTTON ========
            imgui.separator()
            if imgui.button("Back"):
                self.show_autoselect_window = False  # Return to main window
            
            imgui.end()


        ########################################################################
        #                              Operation                              #
        ########################################################################
        win_pos_width = 0
        win_pos_height = self.scene_height + self.object_height

        if not self.show_autosave_window:
            # ==========================
            # MAIN OPERATION WINDOW
            # ==========================
            imgui.set_next_window_position(win_pos_width, win_pos_height)
            imgui.set_next_window_size(self.operation_width, self.operation_height)
            imgui.begin("Operation")

            # Section: Camera Operations
            window_width = imgui.get_window_width()
            text_width = imgui.calc_text_size("Camera Operations").x
            text_pos = (window_width - text_width) / 2
            imgui.set_cursor_pos_x(text_pos)
            imgui.text('Camera Operations')
            imgui.separator()

            imgui.set_next_item_width(imgui.get_window_width())
            if imgui.button("Reset"):
                self.reset()

            if self.multi_cam_flag:
                imgui.push_style_color(imgui.COLOR_BUTTON, 0.6, 0.8, 0.6, 1.0)  # Green when clicked
            else:
                imgui.push_style_color(imgui.COLOR_BUTTON, 1.0, 0.0, 0.0, 1.0)  # Red before clicking

            if imgui.button("Multi Camera"):
                self.multi_cam_flag = not self.multi_cam_flag

            imgui.pop_style_color()

            imgui.set_next_item_width(imgui.get_window_width())
            if imgui.button("Move Camera Around"):
                self.move_camera_flag = not self.move_camera_flag

            if imgui.button('Segmentation'):
                self.segmentation_flag = not self.segmentation_flag
            # Section: Save Options
            imgui.spacing()
            window_width = imgui.get_window_width()
            text_width = imgui.calc_text_size("Save Options").x
            text_pos = (window_width - text_width) / 2
            imgui.set_cursor_pos_x(text_pos)
            imgui.text('Save Options')
            imgui.separator()

            imgui.get_style().colors[imgui.COLOR_BUTTON_HOVERED] = imgui.Vec4(0.6, 0.8, 0.6, 1.0)  # Green hover color

            # ============== AutoSave Button ==============
            imgui.set_next_item_width(100)
            if imgui.button("AutoSave"):
                self.show_autosave_window = True  # Switch to AutoSave window

            # Save RGB Button with Icon
            imgui.set_next_item_width(100)
            save_rgb = self.button_with_icon('icons/save.png', 'Save RGB')
            if save_rgb:
                self.rgb_save_path = self.select_folder()
                if self.rgb_save_path:
                    self.save_rgb(self.rgb_save_path, 0)

            imgui.same_line()

            # Save Depth Button with Icon
            imgui.set_next_item_width(100)
            save_depth = self.button_with_icon('icons/save.png', 'Save Depth')
            if save_depth:
                self.depth_save_path = self.select_folder()
                if self.depth_save_path:
                    self.save_depth(self.depth_save_path, 0)

            imgui.end()

        else:
            # ==========================
            # AUTOSAVE CONFIG WINDOW
            # ==========================
            imgui.set_next_window_position(win_pos_width, win_pos_height)
            imgui.set_next_window_size(self.operation_width, self.operation_height)
            imgui.begin("AutoSave Config")

            imgui.spacing()
            window_width = imgui.get_window_width()
            text_width = imgui.calc_text_size("AutoSave Configs").x
            text_pos = (window_width - text_width) / 2
            imgui.set_cursor_pos_x(text_pos)
            imgui.text('AutoSave Configs')
            imgui.separator()

            imgui.push_item_width(50)
            if imgui.button("RGB Path"):
                self.rgb_save_path = self.select_folder()

            imgui.same_line()
            if imgui.button("Depth Path"):
                self.depth_save_path = self.select_folder()

            imgui.set_next_item_width(100)
            combo_changed, self.layout_opts = imgui.input_int("Layout Opts", self.layout_opts)

            imgui.set_next_item_width(100)
            if imgui.begin_combo("Select Random Options", "Options"):
                # Add checkboxes inside the combo
                _, self.obj_location_option = imgui.checkbox("Object Location", self.obj_location_option)
                _, self.obj_rotation_option = imgui.checkbox("Object Rotation", self.obj_rotation_option)
                _, self.obj_scale_option = imgui.checkbox("Object Scaling", self.obj_scale_option)
                _, self.obj_texture_option = imgui.checkbox("Object Texture", self.obj_texture_option)
                _, self.camera_pos_option = imgui.checkbox("Camera Position", self.camera_pos_option)
                imgui.end_combo()

            if imgui.button("Confirm"):
                if not self.rgb_save_path or not self.depth_save_path:
                    scene_name = self.selected_scene.name
                    self.rgb_save_path = Path("./dataset/rgb") / scene_name
                    self.depth_save_path = Path("./dataset/depth") / scene_name

                    # Create directories if they don't exist
                    self.rgb_save_path.mkdir(parents=True, exist_ok=True)
                    self.depth_save_path.mkdir(parents=True, exist_ok=True)

                self.autosave(self.layout_opts)

            imgui.spacing()
            
            # ======= RETURN TO MAIN WINDOW BUTTON ========
            imgui.separator()
            if imgui.button("Back"):
                self.show_autosave_window = False  # Return to main window
            
            imgui.end()

        ########################################################################
        #                          Camera Configuration                        #
        ########################################################################
        win_pos_width = self.win_width - self.light_config_width
        win_pos_height = 0
        imgui.set_next_window_position(win_pos_width, win_pos_height)
        imgui.set_next_window_size(self.light_config_width, self.light_config_height)
        imgui.begin("Camera Configuration")

        imgui.set_next_item_width(100)
        cam_speed_changed, self.cameraSpeed = imgui.input_float("Camera speed", self.cameraSpeed, format="%.2f")

        imgui.set_next_item_width(100)
        radius_changed, self.sphere_radius = imgui.input_float("Sphere Radius", self.sphere_radius, step=0.1, format="%.2f")
        
        imgui.set_next_item_width(100)
        self.num_vcameras_changed, self.num_vcameras = imgui.input_int("Num of Cameras", self.num_vcameras)

        imgui.set_next_item_width(imgui.get_window_width())
        default_camera = self.button_with_icon('icons/camera.png', 'Default Camera')
        # Switch back to default camera view
        if default_camera:
            self.trackball.set_default()
            self.process_scene_config()
            self.selected_camera = None

        if not self.multi_cam_flag:
            imgui.text(f'Multi-camera is not chosen')
        else: 
            imgui.text(f'Choose a virtual camera')

        imgui.set_next_item_width(100)
        current_item = "" if self.selected_camera is None else self.selected_camera.label
        if imgui.begin_combo("List Cameras", current_item):
            # Iterate through all vcamera
            for i, vcamera in enumerate(self.vcameras):
                if isinstance(vcamera, VCamera):  
                    is_selected = self.selected_camera == vcamera
                    # Render each camera as a selectable item
                    if imgui.selectable(vcamera.label, is_selected)[0]:  
                        self.selected_camera = vcamera
                    # Set the selected state in the combo box
                    if is_selected:
                        imgui.set_item_default_focus()
            imgui.end_combo()
        
        imgui.set_next_item_width(imgui.get_window_width()//1.5)
        imgui.text('Camera Position')
        self.cameraPos_changed, new_camera_pos = imgui.input_float3('', self.cameraPos.x, self.cameraPos.y, self.cameraPos.z, format='%.2f')

        # Convert tuple to glm.vec3
        if self.cameraPos_changed:
            self.old_cameraPos = glm.vec3(self.cameraPos)
            self.cameraPos = glm.vec3(*new_camera_pos)
            self.trackball.update_cameraPos(self.cameraPos, self.old_cameraPos)

        # if imgui.button("Move Camera System"):
        #     if self.multi_cam_flag:
        #         self.move_cam_sys_flag = not self.move_cam_sys_flag # Toggle the flag

        #         hand_cursor = glfw.create_standard_cursor(glfw.HAND_CURSOR)
        #         arrow_cursor = glfw.create_standard_cursor(glfw.ARROW_CURSOR)

        #         if self.move_cam_sys_flag:
        #             # Set the hand cursor for the window
        #             glfw.set_cursor(self.win, hand_cursor)
        #         else:
        #             # Set the hand cursor for the window
        #             glfw.set_cursor(self.win, arrow_cursor)
        #     else:
        #         self.move_cam_sys_flag = False

        # if not self.move_cam_sys_flag:
        #     imgui.text(f'Please turn on multi cam')

        imgui.end()

        ########################################################################
        #                          Light Configuration                         #
        ########################################################################
        win_pos_width = self.win_width - self.light_config_width
        win_pos_height = self.camera_config_height
        imgui.set_next_window_position(win_pos_width, win_pos_height)
        imgui.set_next_window_size(self.light_config_width, self.light_config_height)
        imgui.begin("Light Configuration")

        # Add light position slider
        imgui.set_next_item_width(imgui.get_window_width()//1.5)
        self.lightPos_changed, self.lightPos = imgui.input_float3('Position', self.lightPos[0], self.lightPos[1], self.lightPos[2], format='%.2f')

        # Add light color slider
        imgui.set_next_item_width(imgui.get_window_width()//1.5)
        self.lightColor_changed, self.lightColor = imgui.input_float3('Color', self.lightColor[0], self.lightColor[1], self.lightColor[2], format='%.2f')

        # Add shininess input
        imgui.set_next_item_width(100)
        self.shininess_changed, self.shininess = imgui.input_float("Shininess", self.shininess, step=0.1, format="%.2f")

        imgui.end()

        ########################################################################
        #                              Depth Config                            #
        ########################################################################
        win_pos_width = self.win_width - self.depth_config_width
        win_pos_height = self.camera_config_height + self.light_config_height
        imgui.set_next_window_position(win_pos_width, win_pos_height)
        imgui.set_next_window_size(self.depth_config_width, self.depth_config_height)
        imgui.begin("Depth Config")

        # Center the text "Colormaps for Depth Map"
        window_width = imgui.get_window_width()
        text_width = imgui.calc_text_size("Colormaps for Depth Map").x
        text_pos = (window_width - text_width) / 2 
        imgui.set_cursor_pos_x(text_pos)  
        imgui.text('Colormaps for Depth Map')

        # Get the item spacing from the current style
        style = imgui.get_style()
        spacing = style.item_spacing.x

        # Define button widths to split the window into two halves
        button_width = window_width / 2 - spacing / 2

        # Define colormap options
        if imgui.button('Gray', width=button_width):
            self.selected_colormap = 0
        
        imgui.same_line()
        if imgui.button('Magma', width=button_width):
            self.selected_colormap = 1
        
        # Add near, far 
        self.near_changed, self.near = imgui.input_float("Near", self.near, step=0.1, format="%.2f")
        self.far_changed, self.far = imgui.input_float("Far", self.far, step=0.1, format="%.2f")

        imgui.end()
        imgui.render()
        self.imgui_impl.render(imgui.get_draw_data())

    ''' Magic here '''
    def cam_proj_matrix(self, x_0, y_0, w, h, near, far):
        p00 = 2*self.f_x/w
        p01 = -2*self.s/w
        p02 = (-2*self.c_x + w + 2*x_0)/w
        p11 = -2*self.f_y/h
        p12 = (-2*self.c_y + h + 2*y_0)/h
        p22 = -(far + near)/(far - near)
        p23 = -(2*far*near)/(far - near)
        return np.array(
            [
                [p00, p01, p02, 0  ],
                [0  , p11, p12, 0  ],
                [0  , 0  , p22, p23],
                [0  , 0  , -1 , 0  ]
            ], dtype=np.float32
        )

    ''' Main Loop'''
    def run(self):
        while not glfw.window_should_close(self.win):
            if not self.autosave_flag:
                self.render()

            # Use to debug and find good view for each scene
            # print('track ball cameraPos: ', self.trackball.get_cameraPos())
            self.imgui_menu()

            glfw.swap_buffers(self.win)
            glfw.poll_events()
            self.imgui_impl.process_inputs()

        self.imgui_impl.shutdown()