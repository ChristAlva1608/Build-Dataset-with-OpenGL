import glfw
from shape.viewer import *

def main():
    viewer = Viewer(1918, 480) # NYU resolution
    # viewer = Viewer(1200, 700)  # debug resolution
    viewer.run()

if __name__ == '__main__':
    glfw.init()
    main()
    glfw.terminate()