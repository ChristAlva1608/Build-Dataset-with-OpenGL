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

class Object3D:
    def __init__(self, vert_shader, frag_shader, vert, teco , material, img_path):
        self.vertices = np.array(vert, dtype=np.float32)
        self.textcoords = np.array(teco, dtype=np.float32)

        self.shader = Shader(vert_shader, frag_shader)
        self.vao = VAO()
        self.uma = UManager(self.shader)

        image_path = img_path
        print("Map Kd after passing", material['map_Kd'])
        if material['map_Kd'] is not None:
            print(material['map_Kd'])
            image_path = material['map_Kd']


        image = Image.open(image_path)
        image = image.transpose(Image.FLIP_TOP_BOTTOM)  # Flip image for OpenGL
        img_data = np.array(image, dtype=np.uint8)

        # Generate texture ID and bind it
        self.texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        # Set texture parameters
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        # Load the texture into OpenGL
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image.width, image.height, 0, GL_RGB, GL_UNSIGNED_BYTE, img_data)

    def setup(self):
        self.vao.add_vbo(0, self.vertices, ncomponents=3, dtype=GL.GL_FLOAT, normalized=False, stride=0, offset=None)
        self.vao.add_vbo(3, self.textcoords, ncomponents=3, dtype=GL.GL_FLOAT, normalized=False, stride=0, offset=None)

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

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        glUniform1i(glGetUniformLocation(self.shader.render_idx, "texture1"), 0)

        self.uma.upload_uniform_matrix4fv(np.array(model), 'model', True)
        self.uma.upload_uniform_matrix4fv(np.array(view), 'view', True)
        self.uma.upload_uniform_matrix4fv(np.array(projection), 'projection', True)

        glDrawArrays(GL.GL_TRIANGLES, 0, len(self.vertices)*3)
        self.vao.deactivate()

class Viewer:
    """ GLFW viewer windows, with classic initialization & graphics loop """
    def __init__(self, width=640, height=480):
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

            for drawable in self.drawables:
                if hasattr(drawable, 'key_handler'):
                    drawable.key_handler(key)

    def run(self):
        """ Main render loop for this OpenGL windows """
        while not glfw.window_should_close(self.win):
            # clear draw buffer
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            win_size = glfw.get_window_size(self.win)
            self.model=glm.mat4(1.0)
            self.view = self.trackball.view_matrix()
            self.projection = self.trackball.projection_matrix(win_size)

            # draw our scene objects
            for drawable in self.drawables:
                drawable.draw(self.model, self.view, self.projection)

            glfw.swap_buffers(self.win)
            # Poll for and process events
            glfw.poll_events()

    def add(self, *drawables):
        """ add objects to draw in this windows """
        self.drawables.extend(drawables)

def parse_obj_file(obj_path):
    vertices_all = []
    texcoords_all = []

    # read the first time for vertices and texcoords only
    with open(obj_path, 'r') as f:  # Use 'r' for text files
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):  # Skip empty lines and comments
                continue

            parts = line.split()
            if parts[0] == 'v':  # Vertex definition
                # Parse the vertex coordinates (x, y, z)
                vertices_all.append([float(parts[1]), float(parts[2]), float(parts[3])])

            elif parts[0] == 'vt':  # Texture coordinate definition
                # Parse the texture coordinates (u, v, [w])
                u = float(parts[1])
                v = float(parts[2])
                w = float(parts[3]) if len(parts) > 3 else 0.0  # Default w to 0.0 if not provided
                texcoords_all.append([u, v, w])

    objects = []  # List to hold all the objects

    with open(obj_path, 'r') as f:  # Open the .obj file for reading
        current_obj = None  # Will store the current object being processed

        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):  # Skip empty lines and comments
                continue

            parts = line.split()

            # When encountering a new object (o)
            if parts[0] == 'o':
                pass
            # Process texture name (use 'usemtl' to get texture name if available)
            elif parts[0] == 'usemtl':
                # Start a new object
                current_obj = {
                    'obj_name': parts[1],  # Object name from 'o'
                    'vert_obj_id': [],  # List to store the vertices for the object
                    'textcoords_obj_id': [],  # List to store texture coordinates
                    'texture_name': None,  # Texture name (if any)
                }

                if current_obj:  # If there's an object being processed, add it to the list
                    objects.append(current_obj)



                current_obj['texture_name'] = parts[1]

            # Process faces (f)
            elif parts[0] == 'f':  # Face definition
                # Parse face indices (converting from 1-based to 0-based)
                vert_id = [int(part.split('/')[0]) - 1 for part in parts[1:]]
                text_id = [int(part.split('/')[1]) - 1 for part in parts[1:]]

                if len(vert_id) == 3:
                    current_obj['vert_obj_id'].append(vert_id)
                    current_obj['textcoords_obj_id'].append(text_id)
                elif len(vert_id) == 4:  # Quad face
                    current_obj['vert_obj_id'].append([vert_id[0], vert_id[1], vert_id[2]])  # First triangle
                    current_obj['vert_obj_id'].append([vert_id[0], vert_id[2], vert_id[3]])  # Second triangle

                    current_obj['textcoords_obj_id'].append([text_id[0], text_id[1], text_id[2]])
                    current_obj['textcoords_obj_id'].append([text_id[0], text_id[2], text_id[3]])

        # Append the last object after finishing the file
        if current_obj:
            objects.append(current_obj)

    return vertices_all, texcoords_all, objects

def load_materials(file_path):
    materials = {}
    current_material = None

    def parse_vector(values):
        # Convert space-separated string of numbers into list of floats
        return [float(x) for x in values.strip().split()]

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.strip().split(maxsplit=1)
            if len(parts) < 2:
                continue

            keyword, value = parts

            if keyword == 'newmtl':
                current_material = value
                materials[current_material] = {
                    'Ns': None,  # Specular exponent
                    'Ni': None,  # Optical density
                    'd': None,  # Dissolve
                    'Tr': None,  # Transparency
                    'Tf': None,  # Transmission filter
                    'illum': None,  # Illumination model
                    'Ka': None,  # Ambient color
                    'Kd': None,  # Diffuse color
                    'Ks': None,  # Specular color
                    'Ke': None,  # Emissive color
                    'map_Ka': None,  # Ambient texture
                    'map_Kd': None,  # Diffuse texture
                    'map_bump': None  # Bump map
                }

            elif current_material is not None:
                mat = materials[current_material]

                if keyword == 'Ns':
                    mat['Ns'] = float(value)
                elif keyword == 'Ni':
                    mat['Ni'] = float(value)
                elif keyword == 'd':
                    mat['d'] = float(value)
                elif keyword == 'Tr':
                    mat['Tr'] = float(value)
                elif keyword == 'Tf':
                    mat['Tf'] = parse_vector(value)
                elif keyword == 'illum':
                    mat['illum'] = int(value)
                elif keyword == 'Ka':
                    mat['Ka'] = parse_vector(value)
                elif keyword == 'Kd':
                    mat['Kd'] = parse_vector(value)
                elif keyword == 'Ks':
                    mat['Ks'] = parse_vector(value)
                elif keyword == 'Ke':
                    mat['Ke'] = parse_vector(value)
                elif keyword == 'map_Ka':
                    # Handle potential bump map parameters
                    parts = value.split()
                    mat['map_Ka'] = parts[-1]
                elif keyword == 'map_Kd':
                    parts = value.split()
                    mat['map_Kd'] = parts[-1]
                elif keyword == 'map_bump':
                    # Handle bump map with potential -bm parameter
                    parts = value.split()
                    if '-bm' in parts:
                        bm_index = parts.index('-bm')
                        # Store both the bump multiplier and filename
                        mat['map_bump'] = {
                            'multiplier': float(parts[bm_index + 1]),
                            'filename': parts[-1]
                        }
                    else:
                        mat['map_bump'] = {'filename': parts[-1], 'multiplier': 1.0}

    return materials

vertex_shader_src = "./shader/phong_duck.vert"
fragment_shader_src = "./shader/phong_duck.frag"

# mtl_file = "./obj/house_interior/house_interior.mtl"
# obj_file = "./obj/house_interior/house_interior.obj"

mtl_file = "./obj/church/Sacrification_Church_of_Pyhamaa_SF_2.mtl"
obj_file = "./obj/church/Sacrification_Church_of_Pyhamaa_SF.obj"

img_path = "./image/ledinh.jpeg"

def main():
    vertices_all, texcoords_all, objects = parse_obj_file(obj_file)
    materials = load_materials(mtl_file)
    viewer = Viewer()
    print(len(objects))

    # for key, value in materials.items():
    #     print(value["map_Kd"])

    # print(materials)
    #
    # for obj in objects:
    #     print(obj["texture_name"])

    for obj in objects:
        vertices = []
        tecos = []

        for sublist in obj['vert_obj_id']: # [[1 2 3][2 3 4 ]]
            for vert_id in sublist:
                vertices.append(vertices_all[int(vert_id)])

        for sublist in obj['textcoords_obj_id']:
            for teco_id in sublist:
                tecos.append(texcoords_all[int(teco_id)])

        print(materials[obj['texture_name']])

        model = Object3D(vertex_shader_src,
                         fragment_shader_src,
                         vertices,
                         tecos,
                         materials[obj['texture_name']],
                         img_path
                         )

        model.setup()

        viewer.add(model)

    viewer.run()

glfw.init()
main()
glfw.terminate()


