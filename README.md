# OpenGL-based Data Generation Application

## Abstract
This application is built using OpenGL, incorporating principles of computer graphics and data augmentation techniques. By bridging computer graphics and computer vision camera models, the application enables the generation of high-quality ground truth for computer vision tasks.

## Technology Stack
- **Programming Language**: Python
- **Graphics Library**: PyOpenGL – for OpenGL rendering and graphics logic
- **3D Model Loader**: PyWavefront – to load .obj 3D model files
- **GUI Framework**: ImGui – for interactive user interface overlays
- **Windowing and Input Handling**: GLFW – for window creation and OpenGL context management
- **Math Library**: NumPy / GLM – for vector and matrix operations

## Getting Started
### Prerequisites
- python 3.11+
- pip (Python packages installer)

### Usage
Clone the repository:

```bash
git clone https://github.com/ChristAlva1608/Build-Dataset-with-OpenGL.git
cd Build-Dataset-with-OpenGL
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Place 3D model files (.obj, textures, etc.) in the objects/ and scenes/ directory.

Run in terminal (for MacOS/Linux)
```bash
export PYTHONPATH=./
```

Run the application:
```bash
python main.py
```

To automatically generate a dataset, you must set up some configurations in run.sh and run it
- For MacOS/Linux
```bash
./run.sh
```

- For Window

Control the interface:
- Click **Load Scene** to load a scene
- Click **Load Object** to load objects
- Use ImGui menu to adjust scene, object, camera or depth configurations.

Rendered frames will be saved to the output/ directory.
  
## Project Structure
```
Build-Dataset-with-OpenGL/
├── objects/  # store 3D models for objects
├── scenes/   # store 3D models for scenes
├── shader/   # include vertex shaders and fragment shaders
├── shape/    # include viewer.py which handle main logic for computer graphics
├── main.py   # run this file to open the app
└── run.sh    # run this file to automatically generate a dataset
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
