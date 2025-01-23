import OpenGL.GL as GL
import glfw
import numpy as np
import time
import glm
from itertools import cycle
from PyQt6.QtWidgets import QApplication, QFileDialog
import sys
import time
import imgui
import random
from imgui.integrations.glfw import GlfwRenderer

from libs.buffer import *
from libs.camera import *
from libs.shader import *
from libs.transform import *

from .object3D import *
from .scene3D import *
# from scene3D_v2 import *
from .quad import *
from .vcamera import *
from .sphere import *
from colormap import *
from utils import *


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
        self.phong_shader = Shader("shader/phong.vert", "shader/phong.frag")
        self.phongex_shader = Shader("shader/phongex.vert", "shader/phongex.frag")
        self.texture_shader = Shader('shader/texture.vert', 'shader/texture.frag')
        self.colormap_shader = Shader('shader/colormap.vert', 'shader/colormap.frag')
        self.depth_texture_shader = Shader('shader/depth_texture.vert', 'shader/depth_texture.frag')
        self.good_shader = Shader('shader/good_shader.vert','shader/good_shader.frag')

        self.load_config_flag = False

        # Initialize mouse parameters
        self.last_x = width / 2
        self.last_y = height / 2
        self.first_mouse = True
        self.left_mouse_pressed = False
        self.yaw = -90.0
        self.pitch = 0.0
        self.fov = 45.0

        # Initialize scene config
        self.selected_scene_path = "No file selected"
        self.selected_scene = None
        self.bg_colors = [0.2, 0.2, 0.2]
        self.bg_changed = False

        # Initialize object config
        self.selected_obj_path = "No file selected"
        self.selected_object = None
        self.scale_changed = False
        self.scale_factor = 1
        self.prev_scale_factor = 1
        self.selected_object = None

        # Initialize operation
        self.move_camera_flag = False
        self.multi_cam_flag = False
        self.time_save = 0.0
        self.time_count = 0.0
        self.rgb_save_path = ""
        self.depth_save_path = ""
        self.show_time_selection = False
        self.current_time = 0.0
        self.last_update_time = 0.0

        # Initialize camera config
        self.sphere_radius = 0.1
        self.cameraSpeed = 1
        self.old_cameraPos = glm.vec3(0.0, 0.0, 10.0)
        self.cameraPos = glm.vec3(0.0, 0.0, 10.0)
        self.cameraFront = glm.vec3(0.0, 0.0, 1.0)
        self.cameraUp = glm.vec3(0.0, 1.0, 0.0)
        self.cameraPos_A = glm.vec3(0.0, 0.0, 0.0)
        self.cameraPos_B = glm.vec3(0.0, 0.0, 0.0)
        self.lastFrame = 0.0
        self.selected_camera = None
        self.num_vcameras_changed = False
        self.num_vcameras = 10

        # Initialize light config
        self.shininess = 100
        self.lightPos = glm.vec3(250, 250, 300)
        self.lightColor = glm.vec3(1.0, 1.0, 1.0)

        # Initialize depth config
        self.near = 0.1
        self.far = 1000
        self.near_colors = [0.0, 0.0, 0.0]
        self.far_colors = [1.0, 1.0, 1.0]
        self.selected_colormap = 0
        
        # Initialize trackball
        self.trackball = Trackball()
        self.mouse = (0, 0)

        # Initialize camera pos flag
        self.initial_pos = False

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

        self.scene_width = self.win_width // 6
        self.scene_height = self.win_height // 3

        self.object_width = self.win_width // 6
        self.object_height = self.win_height // 3

        self.operation_width = self.win_width // 6 
        self.operation_height = self.win_height // 3

        self.camera_config_width = self.win_width // 6 
        self.camera_config_height = self.win_height // 3

        self.light_config_width = self.win_width // 6 
        self.light_config_height = self.win_height // 3

        self.depth_config_width = self.win_width // 6 
        self.depth_config_height = self.win_height // 3

        self.rgb_view_width = (self.win_width - self.scene_width - self.light_config_width) // 2
        self.rgb_view_height = self.win_height

        self.depth_view_width = self.rgb_view_width
        self.depth_view_height = self.rgb_view_height

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
    
    def save_rgb(self, save_path):
        # Create a numpy array to hold the pixel data
        win_pos_width = self.scene_width
        pixels = GL.glReadPixels(win_pos_width, 0, self.rgb_view_width, self.rgb_view_height, GL.GL_RGB, GL.GL_UNSIGNED_BYTE)

        # Convert the pixels into a numpy array
        rgb_image = np.frombuffer(pixels, dtype=np.uint8).reshape((int(self.rgb_view_height), int(self.rgb_view_width), 3))

        # Flip the image vertically (because OpenGL's origin is at the bottom-left corner)
        rgb_image = np.flipud(rgb_image)

        # Convert numpy array (or your image data format) to PIL Image
        rgb_image = Image.fromarray(rgb_image)

        # Create a unique file name using timestamp
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        file_name = f"rgb_image_{timestamp}.png"

        # Save the image to the local directory
        rgb_image.save(save_path + '/' + file_name)
        print(f"Saved rgb image as {file_name}")

    def pass_magma_data(self, shader):
        # Flatten the array into 1D
        array_1d = magma_data.flatten()

        GL.glUseProgram(shader.render_idx)
        location = GL.glGetUniformLocation(shader.render_idx, "magma_data")

        # Pass the 1D array to the shader
        GL.glUniform1fv(location, len(array_1d), array_1d)

    def save_depth(self, save_path):
        # Create a numpy array to hold the pixel data
        win_pos_width = self.scene_width + self.rgb_view_width

        # Read Pixel using GL_RGB
        pixels = GL.glReadPixels(win_pos_width, 0, self.depth_view_width, self.depth_view_height, GL.GL_RGB, GL.GL_SHORT) # return linear depth, not raw depth value
        depth_image = np.frombuffer(pixels, dtype=np.short).reshape((self.depth_view_height, self.depth_view_width, 3))

        # Flip the image vertically (because OpenGL's origin is at the bottom-left corner)
        depth_image = np.flipud(depth_image)

        # Get metric depth value for image
        depth_image = depth_image[:,:,0] # get only 1 channel (gray image)
        np.savetxt('depth.txt',depth_image)

        # To visualize depth map, normalize it
        depth_image = (depth_image - depth_image.min()) / (depth_image.max() - depth_image.min())

        # # Convert numpy array (or your image data format) to PIL Image
        # depth_image = Image.fromarray(depth_image)
        
        # Create a unique file name using timestamp
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        file_name = f"depth_image_{timestamp}.png"

        # Save the image to the selected directory
        depth_image.save(save_path + '/' + file_name)
        print(f"Saved depth image as {file_name}")
    
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
    
    def load_cubemap(faces):
        """
        Loads a cubemap texture from a list of file paths.

        Parameters:
            faces (list of str): List of 6 file paths, one for each face of the cubemap.

        Returns:
            int: The OpenGL texture ID of the cubemap.
        """
        texture_id = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_CUBE_MAP, texture_id)

        for i, face in enumerate(faces):
            try:
                # Open the image and ensure it is in RGB format
                with Image.open(face) as img:
                    img = img.convert("RGB")
                    img_data = img.tobytes()
                    width, height = img.size

                    # Upload the image data to the cubemap
                    GL.glTexImage2D(GL.GL_TEXTURE_CUBE_MAP_POSITIVE_X + i,
                                0, GL.GL_RGB, width, height, 0, GL.GL_RGB, GL.GL_UNSIGNED_BYTE, img_data)
            except Exception as e:
                print(f"Cubemap texture failed to load at path: {face}\nError: {e}")

        # Set cubemap texture parameters
        GL.glTexParameteri(GL.GL_TEXTURE_CUBE_MAP, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_CUBE_MAP, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_CUBE_MAP, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_CUBE_MAP, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_CUBE_MAP, GL.GL_TEXTURE_WRAP_R, GL.GL_CLAMP_TO_EDGE)

        return texture_id

    def setup_camA(self):
        self.cameraPos = self.cameraPos_A
        self.cameraFront = glm.vec3(0.0, 0.0, -1.0)

    def setup_camB(self):
        self.cameraPos = self.cameraPos_B
        self.cameraFront = glm.vec3(1.0, 0.0, 0.0)

    def multi_cam(self):
        # Define the hemisphere of multi-camera
        sphere = Sphere(self.phong_shader).setup()

        # # Find sphere cover all the scene
        # for drawable in self.drawables:
        #     if isinstance(drawable, Scene):
        #         if drawable.max_z > drawable.max_x:
        #             sphere.radius = drawable.max_z + (1 / 3) * (drawable.max_z - drawable.min_z)
        #         else:
        #             sphere.radius = drawable.max_x + (1 / 3) * (drawable.max_x - drawable.min_x)
        #     break

        sphere.radius = self.sphere_radius
        sphere.generate_sphere() # update vertices with new radius
        # sphere.draw()

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
            x = sphere.radius * radius_at_y * np.cos(theta)
            z = sphere.radius * radius_at_y * np.sin(theta)
            y = sphere.radius * y
            
            P = glm.vec3(x, y, z)
            O = glm.vec3(0.0, 0.0, 0.0)
            OP = O - P # direction vector of pyramid

            # First, move the pyramid to the position in sphere
            translation = glm.translate(glm.mat4(1.0), P)

            # Split direction vector into 2 component vector
            y_component_vec = glm.vec3(0.0, OP.y, 0.0)
            xz_component_vec = glm.vec3(OP.x, 0.0, OP.z)

            # Two steps to set the direction of the camera directly look into the sphere center
            # # Step 1: rotate around y-axis
            # init_vec1 = glm.vec3(0.0,0.0,-1.0)
            # dot_product = glm.dot(xz_component_vec, init_vec1)
            # magnitude_xz_component_vec = glm.length(xz_component_vec)
            # magnitude_init_vec1 = glm.length(init_vec1)

            # # Calculate the cosine of the angle
            # cos_theta = dot_product / (magnitude_xz_component_vec * magnitude_init_vec1)

            # # Ensure the cosine is within the valid range for arccos due to floating point precision
            # cos_theta = np.clip(cos_theta, -1.0, 1.0)

            # # Calculate the angle in radians
            # y_angle = np.arccos(cos_theta)

            # if P.x < 0: # If position if left-half sphere
            #     y_axis_rotation = glm.rotate(glm.mat4(1.0), -y_angle, glm.vec3(0.0, 1.0, 0.0))
            # else: # If position if left-half sphere
            #     y_axis_rotation = glm.rotate(glm.mat4(1.0), y_angle, glm.vec3(0.0, 1.0, 0.0))

            # Step 2: rotate around z-axis
            init_vec2 = glm.vec3(1.0,0.0,0.0)
            dot_product = glm.dot(OP, init_vec2)
            magnitude_OP = glm.length(OP)
            magnitude_init_vec2 = glm.length(init_vec2)

            # Calculate the cosine of the angle
            cos_theta = dot_product / (magnitude_OP * magnitude_init_vec2)

            # Ensure the cosine is within the valid range for arccos due to floating point precision
            cos_theta = np.clip(cos_theta, -1.0, 1.0)

            # Calculate the angle in radians
            z_angle = np.arccos(cos_theta)
            z_axis_rotation = glm.rotate(glm.mat4(1.0), glm.radians(z_angle), glm.vec3(0.0, 0.0, 1.0))

            # Apply model matrix for pyramid
            vcamera.model = translation 

            # Set up view matrix for camera
            eye = P
            at = P + glm.normalize(P) # Point in the direction of the outward ray
            up = glm.normalize(glm.vec3(vcamera.model[1]))

            vcamera.view = glm.lookAt(eye, at, up)
            vcamera.projection = glm.perspective(glm.radians(self.fov), self.rgb_view_width / self.rgb_view_height, 0.1, 100.0)

            self.vcameras.append(vcamera)

        # For multi-cam system visualization
        for vcamera in self.vcameras:
            new_vcamera = VCamera(self.phong_shader).setup()
            new_vcamera.model = vcamera.model
            new_vcamera.view = self.trackball.view_matrix4()
            new_vcamera.projection = vcamera.projection
            new_vcamera.draw()

    def process_scene_config(self):
        dir_path = './config/'
        file_path = dir_path + os.path.basename(self.selected_scene_path)[:-3] + 'yaml'
        if os.path.exists(file_path):
            content = read_yaml_file(file_path)
            for key, value in content.items():
                if hasattr(self, key):  
                    setattr(self, key, value)
        else:
            print(f'{file_path} is not found. Use default!')

    def run(self):
        while not glfw.window_should_close(self.win):
            if self.selected_scene_path != "No file selected" and self.load_config_flag:
                self.process_scene_config()
                self.load_config_flag = False

            GL.glClearColor(0.2, 0.2, 0.2, 1.0)
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

            if self.multi_cam_flag:
                self.multi_cam()
                GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL) # return back to normal mode

            for drawable in self.drawables:
                drawable.set_mode(1) # mode for rgb image

                # update light configuration
                if self.lightPos_changed:
                    drawable.update_lightPos(self.lightPos)
                
                if self.lightColor_changed:
                    drawable.update_lightColor(self.lightColor)
                
                if self.shininess_changed:
                    drawable.update_shininess(self.shininess)

                # Initially set ideal camera pos for any scene
                if self.initial_pos:
                    x = drawable.max_x + 1/3 * (drawable.max_x - drawable.min_x)
                    z = drawable.max_z + 1/3 * (drawable.max_z - drawable.min_z)
                    
                    # Set up static camera view A and B in initial position
                    self.cameraPos_A = glm.vec3(0.0, 0.0, z)
                    self.cameraPos_B = glm.vec3(x, 0.0, 0.0)

                    # Set up current camera pos
                    self.old_cameraPos = self.cameraPos
                    self.cameraPos = self.cameraPos_A

                if self.move_camera_flag:
                    self.current_time = glfw.get_time()
                    if self.current_time - self.last_update_time >= 1.0:  # Check if 1 second has passed
                        self.cameraPos = self.move_camera_around()  # Update camera position
                        self.last_update_time = self.current_time    
                # self.move_camera_around()

                if self.selected_object == drawable and self.scale_changed:
                    scale_matrix = scale(self.scale_factor) * scale(1/self.prev_scale_factor) # to scale back before applying new scale
                    self.prev_scale_factor = self.scale_factor
                    drawable.update_attribute('model_matrix', scale_matrix)

                view = self.trackball.view_matrix3(self.cameraPos, self.cameraFront, self.cameraUp)
                if self.selected_camera:
                    view = self.selected_camera.view
                drawable.update_attribute('view_matrix', view)

                projection = glm.perspective(glm.radians(self.fov), self.depth_view_width / self.depth_view_height, self.near, self.far)
                drawable.update_attribute('projection_matrix', projection)

                # Normal rendering
                drawable.draw(self.cameraPos)

            # Viewport for Depth Scene
            win_pos_width = self.scene_width + self.rgb_view_width
            win_pos_height = self.win_height - self.depth_view_height # start from bottom-left
            GL.glViewport(win_pos_width, win_pos_height, self.depth_view_width, self.depth_view_height)
            GL.glScissor(win_pos_width, win_pos_height, self.depth_view_width, self.depth_view_height)
            GL.glClearColor(*self.bg_colors, 1.0)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT)

            for drawable in self.drawables:
                drawable.set_mode(0) # mode for depth map

                # update depth map color
                drawable.update_colormap(self.selected_colormap)

                # Initially set ideal camera pos for any scene
                if self.initial_pos:
                    z = drawable.max_z + 1/3 * (drawable.max_z - drawable.min_z)
                    self.cameraPos = glm.vec3(0.0, 0.0, z)

                if self.move_camera_flag:
                    current_time = glfw.get_time()
                    if current_time - self.last_update_time >= 1.0:  # Check if 1 second has passed
                        self.cameraPos = self.move_camera_around()  # Update camera position
                        self.last_update_time = current_time    
                # self.move_camera_around()

                if self.selected_object == drawable and self.scale_changed:
                    scale_matrix = scale(self.scale_factor) * scale(1/self.prev_scale_factor) # to scale back before apply new scale
                    self.prev_scale_factor = self.scale_factor
                    drawable.update_attribute('model_matrix', scale_matrix)

                view = self.trackball.view_matrix3(self.cameraPos, self.cameraFront, self.cameraUp)
                if self.selected_camera:
                    view = self.selected_camera.view
                drawable.update_attribute('view_matrix', view)

                projection = glm.perspective(glm.radians(self.fov), self.depth_view_width / self.depth_view_height, self.near, self.far)
                drawable.update_attribute('projection_matrix', projection)

                # Depth map rendering
                drawable.update_near_far(self.near, self.far)
                
                # Draw the full object
                drawable.draw(self.cameraPos)

                # Visualize with chosen colormap
                if self.selected_colormap == 1:
                    self.pass_magma_data(self.depth_shader)

            GL.glDisable(GL.GL_SCISSOR_TEST)

            self.initial_pos = False

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
            drawable.view = self.trackball.view_matrix()
            drawable.projection = self.trackball.projection_matrix(win_size)

    def move_camera_around(self):
        ''' 
        Set the master camera on the sphere cover the scene. The radius of the sphere is detemined by max of x / z values + offset
        '''
        sphere = Sphere(self.phong_shader).setup()

        for drawable in self.drawables:
            if isinstance(drawable, Scene):
                if drawable.max_z > drawable.max_x:
                    sphere.radius = drawable.max_z + (1 / 3) * (drawable.max_z - drawable.min_z)
                else:
                    sphere.radius = drawable.max_x + (1 / 3) * (drawable.max_x - drawable.min_x)
            break
        
        sphere.generate_sphere() # update vertices with new radius
        # return self.cameraPos_A

        return random.choice(sphere.vertices)
            
    def reset(self):
        pass

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
            
        imgui.text(f"Selected File: {self.selected_scene_path}")

        imgui.set_next_item_width(imgui.get_window_width())
        camera1 = self.button_with_icon('icons/camera.png', 'Camera A')
        camera2 = self.button_with_icon('icons/camera.png', 'Camera B')

        if camera1:
            self.setup_camA()

        if camera2:
            self.setup_camB()

        # Adjust RGB
        self.bg_changed, self.bg_colors = imgui.input_float3('BG Color', self.bg_colors[0], self.bg_colors[1], self.bg_colors[2], format='%.2f')

        font_size = imgui.get_font_size()
        vertical_padding = 8
        button_height = font_size + vertical_padding*2
        imgui.set_cursor_pos((imgui.get_window_width()//4, imgui.get_window_height() - button_height))
        imgui.set_next_item_width(imgui.get_window_width()//2)
        if imgui.button("Load Scene"):
            self.load_config_flag = True
            self.load_scene()

        imgui.end()

        ########################################################################
        #                                Object                                #
        ########################################################################
        
        win_pos_width = 0
        win_pos_height = self.scene_height
        imgui.set_next_window_position(win_pos_width, win_pos_height)
        imgui.set_next_window_size(self.object_width, self.object_height)

        imgui.begin("Object")

        select_obj_flag = self.button_with_icon('icons/load.png', 'Select Obj')
        if select_obj_flag:
            self.selected_obj_path = self.select_file('./object')
            self.load_object()
        imgui.text(f"Selected File: {self.selected_obj_path}")

        imgui.set_next_item_width(100)
        current_item = "" if self.selected_object is None else self.selected_object.name
        if imgui.begin_combo("List of Objects", current_item):
            # Iterate through all drawables
            for drawable in self.drawables:
                if isinstance(drawable, Object):  # Check if the drawable is an Object
                    is_selected = self.selected_object == drawable
                    # Render each object as a selectable item
                    if imgui.selectable(drawable.name, is_selected)[0]:  # Note: selectable returns a tuple
                        self.selected_object = drawable
                    # Set the selected state in the combo box
                    if is_selected:
                        imgui.set_item_default_focus()
            imgui.end_combo()

        if self.selected_object:
            imgui.text(f"{self.selected_object.name} is selected")
        else:
            imgui.text(f"No object is selected")

        imgui.set_next_item_width(100)
        self.scale_changed, self.scale_factor = imgui.input_float("Scale factor", self.scale_factor, format="%.2f")

        if imgui.button("Drag/Drop"):
            # Create a hand cursor
            hand_cursor = glfw.create_standard_cursor(glfw.HAND_CURSOR)

            # Set the hand cursor for the window
            glfw.set_cursor(self.win, hand_cursor)

            self.drag_object() # TO-DO CODE

        font_size = imgui.get_font_size()
        vertical_padding = 8
        button_height = font_size + vertical_padding*2
        imgui.set_cursor_pos((imgui.get_window_width()//4, imgui.get_window_height() - button_height))
        imgui.set_next_item_width(imgui.get_window_width()//2)
        if imgui.button("Load Object"):
            self.load_object()

        imgui.end()

        ########################################################################
        #                              Operation                              #
        ########################################################################
        win_pos_width = 0
        win_pos_height = self.scene_height + self.object_height
        imgui.set_next_window_position(win_pos_width, win_pos_height)
        imgui.set_next_window_size(self.operation_width, self.operation_height)
        imgui.begin("Operation")

        imgui.set_next_item_width(imgui.get_window_width())
        if imgui.button("Use Trackball"):
            self.use_trackball()

        imgui.set_next_item_width(imgui.get_window_width())
        if imgui.button("Multi Camera"):
            self.multi_cam_flag = True

        imgui.set_next_item_width(imgui.get_window_width())
        if imgui.button("Reset"):
            self.reset()

        imgui.get_style().colors[imgui.COLOR_BUTTON_HOVERED] = imgui.Vec4(0.6, 0.8, 0.6, 1.0)  # Green hover color
        imgui.set_next_item_width(100)
        if imgui.button("AutoSave"):
            self.show_time_selection = True

        # imgui.same_line()
        imgui.set_next_item_width(100)
        save_rgb = self.button_with_icon('icons/save.png', 'Save RGB')
        if save_rgb:
            self.rgb_save_path = self.select_folder()
            self.save_rgb(self.rgb_save_path)

        imgui.same_line()
        imgui.set_next_item_width(100)
        save_depth = self.button_with_icon('icons/save.png', 'Save Depth')
        if save_depth:
            self.depth_save_path = self.select_folder()
            self.save_depth(self.depth_save_path)

        if self.show_time_selection:
            imgui.push_item_width(50)
            
            if imgui.button("RGB Save Path"):
                self.rgb_save_path = self.select_folder()

            imgui.same_line()
            if imgui.button("Depth Save Path"):
                self.depth_save_path = self.select_folder()

            time_changed, self.time_save = imgui.input_float("Time Selection (s)", self.time_save)
            if time_changed:
                self.time_count = time.time() # reset time count if time selection is changed

        if ((time.time() - self.time_count >= self.time_save) and (self.time_save != 0)): # make a loop of autosave
            self.save_rgb(self.rgb_save_path)
            self.save_depth(self.depth_save_path)
            self.time_count = time.time()

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
        radius_changed, radius_value = imgui.slider_float("Sphere Radius",
                                          self.sphere_radius,
                                          min_value=0.00,
                                          max_value=500.00,
                                          format="%.2f")
        
        if radius_changed:
            self.sphere_radius = radius_value
        
        imgui.set_next_item_width(100)
        self.num_vcameras_changed, self.num_vcameras = imgui.input_int("Num of Cameras", self.num_vcameras)

        imgui.set_next_item_width(imgui.get_window_width())
        default_camera = self.button_with_icon('icons/camera.png', 'Default Camera')
        # Switch back to default camera view
        if default_camera:
            self.initial_pos = True
            self.selected_camera = None

        if not self.multi_cam_flag:
            imgui.text(f'Multi-camera is not been chosen')
        else: 
            imgui.text(f'Please choose a virtual camera')

        imgui.set_next_item_width(100)
        current_item = "" if self.selected_camera is None else self.selected_camera.label
        if imgui.begin_combo("List of Cameras", current_item):
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
        
        if radius_changed:
            self.radius = radius_value
        
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

        # Add shininess slider
        imgui.set_next_item_width(imgui.get_window_width()//2)
        self.shininess_changed, shininess_value = imgui.slider_float("Shininess",
                                          self.shininess,
                                          min_value=0.00,
                                          max_value=100.00,
                                          format="%.2f")
        if self.shininess_changed:
            self.shininess = shininess_value

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

        # imgui.set_next_item_width(100)
        self.near_changed, near_value = imgui.slider_float("Near",
                                          self.near,
                                          min_value=0.1,
                                          max_value=10,
                                          format="%.1f"
                                          )
        if self.near_changed:
            self.near = near_value

        # Add far plane slider
        # imgui.set_next_item_width(100)
        self.far_changed, far_value = imgui.slider_int("Far",
                                          int(self.far),
                                          min_value=1,
                                          max_value=5000
                                          )
        if self.far_changed:
            self.far = far_value

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
    
    def reset(self):
        # Remove all scenes and objects
        self.drawables.clear()

    def lay_object(self):
        if not self.selected_object or not self.selected_scene:
            return glm.mat4(1.0)
        object_bottom = self.selected_object.min_y
        scene_bottom = self.selected_scene.min_y

        if object_bottom > scene_bottom and scene_bottom < 0:
            translation_matrix = translate(0, (scene_bottom - object_bottom), 0)
        if object_bottom < scene_bottom and scene_bottom < 0:
            translation_matrix = translate(0, -(scene_bottom - object_bottom), 0)
        if object_bottom > scene_bottom and scene_bottom > 0:
            translation_matrix = translate(0, -(scene_bottom - object_bottom), 0)
        if object_bottom < scene_bottom and scene_bottom > 0:
            translation_matrix = translate(0, (scene_bottom - object_bottom), 0)

        return translation_matrix

    def load_scene(self):
        model = []

        # remove last Scene
        self.drawables = [drawable for drawable in self.drawables if not isinstance(drawable, Scene)]

        # Add chosen object or scene
        if self.selected_scene_path != "No file selected":
            self.initial_pos = True # Initially set up camera pos for the scene
            self.selected_scene = Scene(self.depth_texture_shader, self.selected_scene_path)
            model.append(self.selected_scene)

        self.add(model)

    def load_object(self):
        model = []

        if self.selected_obj_path != "No file selected":
            self.selected_object = Object(self.depth_texture_shader, self.selected_obj_path)
            # translation_matrix = self.lay_object()
            # self.selected_object.update_attribute('model_matrix', translation_matrix)
            model.append(self.selected_object)

        self.add(model)