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

from model3D import *
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
        self.texture_shader = Shader("shader/texture.vert", "shader/texture.frag")
        self.main_shader = Shader("shader/phong.vert", "shader/phong.frag")
        self.phongex_shader = Shader("shader/phongex.vert", "shader/phongex.frag")
        
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
        self.selected_vcamera = -1
        self.multi_camera_option = False
        self.single_camera_option = False

        # Initialize trackball
        self.trackball = Trackball()
        self.mouse = (0, 0)
        
        # Initialize vcamera parameters
        self.cameraSpeed = 0.5
        self.cameraPos = glm.vec3(0.0, 0.0, 0.0)   
        self.cameraFront = glm.vec3(0.0, 0.0, -1.0)  
        self.cameraUp = glm.vec3(0.0, 1.0, 0.0)    
        self.lastFrame = 0.0

        # Initialize sphere
        self.sphere_radius = 8.0

        # Initiliaze for Viewport
        self.rows = 3
        self.cols = 3
        self.left_width = self.win_width
        self.left_height = self.win_height*2
        self.right_width = self.win_width
        self.right_height = self.win_height*2
        self.cell_width = self.right_width // self.cols
        self.cell_height = self.right_height // self.rows

        # Initialize for camera 
        self.camera1_cameraPos = glm.vec3(0.0, 0.0, 0.0)
        self.camera2_cameraPos = glm.vec3(5.0, 0.0, 0.0)
        
        # Initialize background color
        self.bg_colors = [0.0, 0.0, 0.0]
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
        # # Initialize item for "Scene"
        # self.scene_items = [
        #     {"text": f"Clickable Text Item", "function": self.save_rgb, "hovered": False, "clicked": False}


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
        self.rgb_view_height = 2 * self.win_height // 3

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
    
    def multi_cam(self):
        # Set up some parameters
        rows = 3
        cols = 3
        left_width = 1400
        left_height = 700
        right_width = 1400
        right_height = 400
        cell_width = right_width // cols
        cell_height = right_height // rows

        # Define the hemisphere of multi-camera
        sphere = Sphere(self.main_shader).setup()
        # sphere.radius = 4.0
        # sphere.generate_sphere()

        ######
        # [[x,y,z],[x,y,z]]
        ######
        self.vcameras = []
        for coord in sphere.vertices[1:len(sphere.vertices)//2:2]:
            initial = coord[0]

            vcamera = VCamera(self.main_shader)
            vcamera.setup()
            
            P = glm.vec3(coord[0], coord[1], coord[2])
            
            # Set up view matrix for camera
            yaw, pitch = self.get_yaw_pitch_from_direction(P, glm.vec3(0, 0, 0))
            direction = glm.vec3(
                np.cos(glm.radians(yaw)) * np.cos(glm.radians(pitch)),
                np.sin(glm.radians(pitch)),
                np.sin(glm.radians(yaw)) * np.cos(glm.radians(pitch))
            )
            right = glm.normalize(glm.cross(direction, glm.vec3(0, 1, 0)))
            up = glm.normalize(glm.cross(right, direction))

            vcamera.model = glm.mat4(1.0)
            vcamera.view = glm.lookAt(P, P + direction, up)
            vcamera.projection = glm.perspective(glm.radians(self.fov), cell_width / cell_height, 0.1, 1000.0)
            
            self.vcameras.append(vcamera)

        # Normal scene
        GL.glViewport(0, 0, left_width, left_height*2)
        GL.glUseProgram(self.main_shader.render_idx)

        for cam in self.vcameras:
            cam.projection = self.trackball.projection_matrix((left_width, left_height*2))
            cam.draw()
        for drawable in self.drawables:
            win_size = (left_width, left_height*2)

            drawable.update_uma(UManager(self.main_shader))
            drawable.update_shader(self.main_shader)
            drawable.setup()

            drawable.view = self.trackball.view_matrix2(self.cameraPos)
            drawable.projection = self.trackball.projection_matrix(win_size)
            drawable.draw()

        # Depth map
        GL.glViewport(left_width, right_height*2, right_width, right_height)
        GL.glUseProgram(self.depth_shader.render_idx)
        for drawable in self.drawables:
            drawable.update_uma(UManager(self.depth_shader))
            drawable.update_shader(self.depth_shader)
            drawable.setup()
            
            drawable.view = self.vcameras[self.selected_vcamera].view 
            drawable.projection = self.vcameras[self.selected_vcamera].projection

            drawable.shader.set_uniform('near', 0.1)
            drawable.shader.set_uniform('far', 1000.0)
            drawable.draw()

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
            # GL.glClearColor(*self.bg_colors, 1.0)
            # print(self.bg_colors)
            # print('----------')
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
            
            # Viewport for RGB Scene
            win_pos_width = self.scene_width + self.operation_width
            win_pos_height = self.win_height - self.rgb_view_height # start from bottom-left
            GL.glViewport(win_pos_width, win_pos_height, self.rgb_view_width, self.rgb_view_height) # Please remove *2 if not using MacOS
            # GL.glViewport(0, 0, self.win_width//2, self.win_height)
            GL.glClearColor(1.0, 1.0, 1.0, 1.0)
            GL.glUseProgram(self.main_shader.render_idx)

            for drawable in self.drawables:
                drawable.update_uma(UManager(self.main_shader))
                drawable.update_shader(self.main_shader)
                drawable.setup()

                drawable.model = glm.mat4(1.0)
                # drawable.view = self.trackball.view_matrix2(self.cameraPos)
                drawable.view = self.trackball.view_matrix()
                drawable.projection = glm.perspective(glm.radians(self.fov), self.rgb_view_width / self.rgb_view_height, 0.1, 1000.0)
                
                # Normal rendering
                drawable.draw()

            # Viewport for Depth Scene
            win_pos_width = self.scene_width + self.operation_width + self.rgb_view_width
            win_pos_height = self.win_height - self.depth_view_height # start from bottom-left
            GL.glViewport(win_pos_width, win_pos_height, self.depth_view_width, self.depth_view_height)
            GL.glClearColor(1.0, 1.0, 1.0, 1.0)
            GL.glUseProgram(self.depth_shader.render_idx)
            for drawable in self.drawables:
                drawable.update_uma(UManager(self.depth_shader))
                drawable.update_shader(self.depth_shader)
                drawable.setup()
                
                drawable.model = glm.mat4(1.0)
                drawable.view = self.trackball.view_matrix2(self.cameraPos)
                # drawable.view = self.trackball.view_matrix()
                drawable.projection = glm.perspective(glm.radians(self.fov), self.depth_view_width / self.depth_view_height, 0.1, 1000.0)
                
                # Depth map rendering
                drawable.shader.set_uniform('near', 0.1)
                drawable.shader.set_uniform('far', 1000.0)
                drawable.draw()
            
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

        imgui.set_window_font_scale(1)  
        
        item_width = imgui.get_window_width() - 10 # padding
        imgui.set_next_item_width(item_width)
        _, self.selected_obj = imgui.combo(
            "", 
            int(self.selected_obj),
            ["Wuson", "Porsche", "Bathroom", "Building",
             "Castelia City", "House Interior"]
        )

        imgui.set_next_item_width(imgui.get_window_width())
        camera1 = self.button_with_icon('icons/camera.png', 'Camera A')
        camera2 = self.button_with_icon('icons/camera.png', 'Camera B')

        if camera1:
            self.cameraPos = self.camera1_cameraPos

        if camera2:
            self.cameraPos = self.camera2_cameraPos
        
        # Adjust RGB 
        self.bg_changed, self.bg_colors = imgui.input_int3('Background Color', self.bg_colors[0], self.bg_colors[1], self.bg_colors[2])

        font_size = imgui.get_font_size()
        vertical_padding = 8
        button_height = font_size + vertical_padding*2
        imgui.set_cursor_pos((imgui.get_window_width()//4, imgui.get_window_height() - button_height))
        imgui.set_next_item_width(imgui.get_window_width()//2)
        if imgui.button("Load Model"):
            self.create_model()

        imgui.end()

        ########################################################################
        #                              Operataion                              #
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
        # if self.fov < 1.0:
        #     self.fov = 1.0
        # if self.fov > 45.0:
        #     self.fov = 45.0
        
        self.trackball.zoom(yoffset, glfw.get_window_size(window)[1])

    def create_model(self):
        model = []
        self.drawables.clear()
        
        obj_files = {
            0: 'obj/WusonOBJ.obj',
            1: 'obj/Porsche_911_GT2.obj',
            2: 'obj/bathroom/bathroom.obj',
            3: 'obj/building/Residential Buildings 002.obj',
            4: 'obj/Castelia_City/Castelia_City.obj',
            5: 'obj/house_interior/house_interior.obj'
        }

        chosen_obj = obj_files[self.selected_obj]
        model.append(Obj(chosen_obj))

        self.add(model)

def main():
    viewer = Viewer()
    viewer.run()

if __name__ == '__main__':
    glfw.init()
    main()
    glfw.terminate()