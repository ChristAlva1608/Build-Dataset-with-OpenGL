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
from imgui.integrations.glfw import GlfwRenderer
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


class Viewer:
    def __init__(self, width=1300, height=700):
        self.fill_modes = cycle([GL.GL_LINE, GL.GL_POINT, GL.GL_FILL])
        
        # GLFW initialization
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL.GL_TRUE)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.RESIZABLE, False)
        glfw.window_hint(glfw.DEPTH_BITS, 16)
        glfw.window_hint(glfw.DOUBLEBUFFER, True)
        
        self.win = glfw.create_window(width, height, 'Viewer', None, None)
        if not self.win:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")
        
        glfw.make_context_current(self.win)

        self.width, self.height = glfw.get_framebuffer_size(self.win)
        # Initialize imgui
        imgui.create_context()
        self.imgui_impl = GlfwRenderer(self.win)

        # Enable depth testing
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDepthFunc(GL.GL_LESS)
        
        # Initialize shaders
        self.depth_shader = Shader("shader/depth.vert", "shader/depth.frag")
        self.phong_shader = Shader("shader/phong.vert", "shader/phong.frag")
        self.phongex_shader = Shader("shader/phongex.vert", "shader/phongex.frag")
        self.texture_shader = Shader("shader/texture.vert", "shader/texture.frag")
        
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
        self.left_width = self.width//2
        self.left_height = self.height
        self.right_width = self.width//2
        self.right_height = self.height
        self.cell_width = self.right_width // self.cols
        self.cell_height = self.right_height // self.rows

        # Initialize for vcamera 
        self.num_vcameras = 1
        self.cam_base_width=5
        self.cam_base_height=5
        self.cam_height=3.5

        # Initialize input text
        self.text = ""

        # Register callbacks

        glfw.set_key_callback(self.win, self.on_key)
        glfw.set_cursor_pos_callback(self.win, self.on_mouse_move)
        glfw.set_scroll_callback(self.win, self.scroll_callback)

        print('OpenGL', GL.glGetString(GL.GL_VERSION).decode() + ', GLSL',
              GL.glGetString(GL.GL_SHADING_LANGUAGE_VERSION).decode() +
              ', Renderer', GL.glGetString(GL.GL_RENDERER).decode())

        GL.glClearColor(1.0, 1.0, 1.0, 1.0)
        self.drawables = []

    def save_rgb(self):

        # Create a numpy array to hold the pixel data
        pixels = GL.glReadPixels(0, 0, self.width, self.height*2, GL.GL_RGB, GL.GL_UNSIGNED_BYTE)

        # Convert the pixels into a numpy array
        rgb_image = np.frombuffer(pixels, dtype=np.uint8).reshape((int(self.height*2), int(self.width), 3))

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
        pixels = GL.glReadPixels(self.width, 0, self.width, self.height*2, GL.GL_RGB, GL.GL_UNSIGNED_BYTE)

        # Convert the pixels into a numpy array
        depth_image = np.frombuffer(pixels, dtype=np.uint8).reshape((self.height*2, self.width, 3))

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
        left_width = 1300
        left_height = 700
        right_width = 1300
        right_height = 400
        cell_width = right_width // cols
        cell_height = right_height // rows

        # Define the hemisphere of multi-camera
        sphere = Sphere(self.phong_shader).setup()
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

    def run(self):
        while not glfw.window_should_close(self.win):
            GL.glClearColor(0.2, 0.2, 0.2, 1.0)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

            # if self.diffuse_changed:
            #     for drawable in self.drawables:
            #         drawable.K_materials[0] = self.diffuse

            # if self.specular_changed:
            #     for drawable in self.drawables:
            #         drawable.K_materials[1] = self.specular

            # if self.ambient_changed:
            #     for drawable in self.drawables:
            #         drawable.K_materials[2] = self.ambient


            # if self.multi_camera_option:
            #     # Activate multi vcamera system
            #     self.multi_cam()
            
            if self.single_camera_option:
                # Normal scene
                GL.glViewport(0, 0, self.left_width, self.left_height)
                GL.glUseProgram(self.phong_shader.render_idx)

                for drawable in self.drawables:
                    drawable.update_shader(self.phong_shader)
                    drawable.setup()

                    drawable.model = glm.mat4(1.0)
                    # drawable.view = self.trackball.view_matrix2(self.cameraPos)
                    drawable.view = self.trackball.view_matrix()
                    drawable.projection = glm.perspective(glm.radians(self.fov), 1300.0 / 700.0, 0.1, 1000.0)
                    
                    # Normal rendering
                    drawable.draw()

                # Depth map
                GL.glViewport(self.left_width, 0, self.right_width, self.right_height)
                GL.glUseProgram(self.depth_shader.render_idx)
                for drawable in self.drawables:
                    drawable.update_shader(self.depth_shader)
                    drawable.setup()
                    
                    drawable.model = glm.mat4(1.0)
                    # drawable.view = self.trackball.view_matrix2(self.cameraPos)
                    drawable.view = self.trackball.view_matrix()
                    drawable.projection = glm.perspective(glm.radians(self.fov), 1300.0 / 700.0, 0.1, 1000.0)
                    
                    # Depth map rendering
                    drawable.shader.set_uniform('near', 0.1)
                    drawable.shader.set_uniform('far', 1000.0)
                    drawable.draw()
            
            self.imgui_menu()

            glfw.swap_buffers(self.win)
            glfw.poll_events()
            self.imgui_impl.process_inputs()

        self.imgui_impl.shutdown()

    def imgui_menu(self):
        imgui.new_frame()
        imgui.set_next_window_position(0, 0)
        imgui.set_next_window_size(300, 200)
        imgui.begin("Controls")

        imgui.set_next_item_width(100)
        _, self.selected_obj = imgui.combo(
            "Select Object",
            int(self.selected_obj),
            ["Wuson", "Porsche", "Bathroom", "Building",
             "Castelia City", "House Interior"]
        )

        imgui.set_next_item_width(100)
        if imgui.begin_combo("Select Option", "Options"):
            # Add checkboxes inside the combo
            _, self.single_camera_option = imgui.checkbox("Single Camera", self.single_camera_option)
            _, self.multi_camera_option = imgui.checkbox("Multi Camera", self.multi_camera_option)
            imgui.end_combo()

        # Add Diffuse slider
        imgui.set_next_item_width(100)
        self.diffuse_changed, diffuse_value = imgui.slider_float("Diffuse", 
                                          self.diffuse, 
                                          min_value=0.00, 
                                          max_value=1,
                                          format="%.2f")
        if self.diffuse_changed:
            self.diffuse = diffuse_value

        # Add Ambient slider
        imgui.set_next_item_width(100)
        self.ambient_changed, ambient_value = imgui.slider_float("Ambient", 
                                          self.ambient, 
                                          min_value=0.00, 
                                          max_value=1,
                                          format="%.2f")
        if self.ambient_changed:
            self.ambient = ambient_value

        # Add Specular slider
        imgui.set_next_item_width(100)
        self.specular_changed, specular_value = imgui.slider_float("Specular", 
                                          self.specular, 
                                          min_value=0.00, 
                                          max_value=1,
                                          format="%.2f")
        if self.specular_changed:
            self.specular = specular_value

        if self.multi_camera_option:
            imgui.set_next_item_width(100)
            _, self.selected_vcamera = imgui.combo(
                "Select VCamera",
                int(self.selected_vcamera),
                [str(i) for i in range(1, self.num_vcameras + 1)]
            )

        imgui.set_next_item_width(100)
        if imgui.button("Load Model"):
            self.create_model()

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
        model.append(Obj(self.phongex_shader, chosen_obj))

        self.add(model)

def main():
    viewer = Viewer()
    viewer.run()

if __name__ == '__main__':
    glfw.init()
    main()
    glfw.terminate()