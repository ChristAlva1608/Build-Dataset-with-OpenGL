import glfw
from shape.viewer import *

def main():
    viewer = Viewer(1200, 700)
    viewer.run()

if __name__ == '__main__':
    glfw.init()
    main()
    glfw.terminate()