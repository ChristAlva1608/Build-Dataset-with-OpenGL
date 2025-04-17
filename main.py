import glfw
import argparse 
from shape.viewer import *

def main():
    parser = argparse.ArgumentParser(description="Automate Dataset Creation.")

    # Add arguments
    parser.add_argument("--scene_path", type=str, default='No file selected', help="Path to 3D scene model")
    parser.add_argument("--obj_folder_path", type=str, default='No file selected', help="Path to object storage")
    parser.add_argument("--rgb_save_path", type=str, default='', help="RGB save path")
    parser.add_argument("--depth_save_path", type=str, default='', help="Depth save path")

    parser.add_argument("--num_objects", type=int, default=20, help="Number of objects to load")
    parser.add_argument("--num_images_train", type=int, default=100, help="Number of images to save for train set")
    parser.add_argument("--num_images_test", type=int, default=100, help="Number of images to save for test set")

    parser.add_argument("--obj_location", action="store_true", help="Randomize Object Location")
    parser.add_argument("--obj_rotation", action="store_true", help="Randomize Object Rotation")
    parser.add_argument("--obj_scale", action="store_true", help="Randomize Object Scaling")
    parser.add_argument("--obj_texture", action="store_true", help="Randomize Object Texture")
    parser.add_argument("--camera_pos", action="store_true", help="Randomize Camera Position")

    parser.add_argument("--scene_net_layout", action="store_true", help="switch to SceneNet flow")
    # parser.add_argument("--NYU_texture", action="store_true", help="picking random NYU texture for scene and object")
    parser.add_argument("--NYU_path", type=str, default='something', help="Path to NYU training set (if not empty string then picking random NYU texture for scene and object)")

    # Auto flag: If set, "load_scene" and "autosave" is automatic
    parser.add_argument("--auto", action="store_true", help="Enable automatic execution (skip manual confirmation)")

    # Parse arguments
    args = parser.parse_args()

    viewer = Viewer(args, img_width=640, img_height=480) # NYU resolution
    # viewer = Viewer(args,1200, 700)  # debug resolution
    viewer.run()

if __name__ == '__main__':
    glfw.init()
    main()
    glfw.terminate()