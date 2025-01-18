import pywavefront
import glfw
from libs.shader import *
from libs.transform import *
from libs.buffer import *
from libs.camera import *
from OpenGL.GL import *
from itertools import cycle
from PIL import Image
import glm

def convert_LA_to_RGBA(image):
    if image.mode == "LA":
        # Split the Luminance and Alpha channels
        luminance, alpha = image.split()
        rgb_image = Image.merge("RGB", (luminance, luminance, luminance))
        rgba_image = Image.merge("RGBA", (luminance, luminance, luminance, alpha))
        return rgba_image
    else:
        return image  # Return unchanged for non-LA images

class Object3D:
    def __init__(self, vert_shader, frag_shader, vert, teco, material):

        self.rgb = True
        self.has_texture = True if len(teco) > 0 else False
        self.vertices = np.array(vert, dtype=np.float32)
        self.textcoords = np.array(teco, dtype=np.float32)

        self.shader = Shader(vert_shader, frag_shader)
        self.vao = VAO()
        self.uma = UManager(self.shader)

        # image_path = img_path

        # if material['map_Ka'] is not None:
        #     print(material['map_Ka'])
        #     image_path = material['map_Ka']

        image_path = "./image/ledinh.jpeg"

        if material['map_Kd'] is not None:
            # print(material['map_Kd'])
            image_path = material['map_Kd']
            image_path = './obj/warehouse/' + image_path # hardcode
            print(image_path)

        # if self.has_texture:
        #     print(image_path)
        #     self.uma.setup_texture("texture1", image_path)

        image = Image.open(image_path)

        if image.mode == "LA":
            image = convert_LA_to_RGBA(image)
            self.rgb = False
        elif image.mode == "RGBA":
            self.rgb = False

        image = image.transpose(Image.FLIP_TOP_BOTTOM)  # Flip image for OpenGL
        img_data = np.array(image, dtype=np.uint8)

        # Generate texture ID and bind it
        self.texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        # Set texture parameters
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        # flatColor = [1.0, 1.0, 1.0, 1.0]
        # glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, flatColor)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        # Load the texture into OpenGL
        if self.rgb:
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image.width, image.height, 0, GL_RGB, GL_UNSIGNED_BYTE, img_data)
        else:
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image.width, image.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, img_data)

    def setup(self):
        self.vao.add_vbo(0, self.vertices, ncomponents=3, dtype=GL.GL_FLOAT, normalized=False, stride=0, offset=None)
        self.vao.add_vbo(3, self.textcoords, ncomponents=2, dtype=GL.GL_FLOAT, normalized=False, stride=0, offset=None)

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

        self.uma.upload_uniform_scalar1f(shininess, 'shininess')
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

        glDrawArrays(GL_TRIANGLES, 0, len(self.vertices)*3)

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
        glEnable(GL.GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        # glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA)
        # glEnable(GL_MULTISAMPLE)

        # glCullFace(GL_FRONT)
        # glFrontFace(GL_CCW)

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
            self.projection = self.trackball.projection_matrix(win_size[0]/win_size[1])

            # draw our scene objects
            for drawable in self.drawables:
                drawable.draw(self.model, self.view, self.projection)

            glfw.swap_buffers(self.win)
            # Poll for and process events
            glfw.poll_events()

    def add(self, *drawables):
        """ add objects to draw in this windows """
        self.drawables.extend(drawables)

# obj_file = "./obj/house_interior/house_interior.obj"
# mtl_file = "./obj/house_interior/house_interior.mtl"

# obj_file = "./obj/church/Sacrification_Church_of_Pyhamaa_SF.obj"
# mtl_file = "./obj/church/Sacrification_Church_of_Pyhamaa_SF.mtl"

obj_file = "./obj/warehouse/ware-house.obj"
mtl_file = "./obj/warehouse/ware-house.mtl"

texture_file = "./image/ledinh.jpeg"

vertex_shader_src = "./shader/phong_duck.vert"
fragment_shader_src = "./shader/phong_duck.frag"

def process_material_data(material, current_obj, UV_flag):
    data = material.vertices  # [vt_x1, vt_y1, vn_x1    , vn_y1, vn_z1, v_x1, v_y1, v_z1 ...]
    current_obj['material'] = material.name

    # Define the size of each group
    num_texcoords = 2
    num_normals = 3
    num_vertices = 3

    if UV_flag:
        for i in range(0, len(data), num_texcoords + num_normals + num_vertices):
            current_obj['texcoords'].append(data[i:i + num_texcoords])
            current_obj['normals'].append(data[i + num_texcoords:i + num_texcoords + num_normals])
            current_obj['vertices'].append(
                data[i + num_texcoords + num_normals:i + num_texcoords + num_normals + num_vertices])
    else:
        for i in range(0, len(data), num_normals + num_vertices):
            current_obj['normals'].append(data[i:i + num_normals])
            current_obj['vertices'].append(
                data[i + num_normals:i + num_normals + num_vertices]
            )

def parse_file_pywavefront(obj_file):
    scene = pywavefront.Wavefront(obj_file, collect_faces=True)
    list_mesh = []
    print("Length of mesh items:", len(scene.meshes.items()))

    for mesh_name, mesh in scene.meshes.items():
        print("Length of materials:", len(mesh.materials))
        if len(mesh.materials) == 1: # 1 object 1 material
            print("each object has 1 materials")
            current_obj = {
                'name': mesh_name,
                'texcoords': [],
                'normals': [],
                'vertices': [],
                'material': None,
            }
            # print("Has texture (sig)", mesh.materials[0].has_uvs)
            hasUV = mesh.materials[0].has_uvs
            process_material_data(mesh.materials[0], current_obj, hasUV)

            list_mesh.append(current_obj)

        else: # 1 object many materials
            print("has multiple materials")
            for mat in mesh.materials:
                current_obj = {
                    'name': mat.name,
                    'texcoords': [],
                    'normals': [],
                    'vertices': [],
                    'material': None,
                }
                # print("Has texture (mul)", mat.has_uvs)
                hasUV = mat.has_uvs
                process_material_data(mat, current_obj, hasUV)
                list_mesh.append(current_obj)

    return list_mesh

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

def printout_object(object):
    print("Length vertices:", len(object['vertices']))
    print("Length textcoord:", len(object['texcoords']))
    print("Material name", object['material'])

# start from here
def main():
    listObjects = parse_file_pywavefront(obj_file)
    materials = load_materials(mtl_file)

    viewer = Viewer()

    for obj in listObjects: #
        model = Object3D(vertex_shader_src,
                         fragment_shader_src,
                         obj['vertices'],
                         obj['texcoords'],
                         materials[obj['material']]
                         )
        model.setup()
        viewer.add(model)

    viewer.run()

glfw.init()
main()
glfw.terminate()

