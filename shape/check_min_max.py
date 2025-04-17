# this file use for checking the min max vertices of the obj file to config the .yaml file
import os

def check_min_max(scene_path):
    min_x = min_y = min_z = float('inf')
    max_x = max_y = max_z = float('-inf')

    with open(scene_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if parts and parts[0] == 'v':  # Only process vertex lines
                x, y, z = map(float, parts[1:4])
                min_x, max_x = min(min_x, x), max(max_x, x)
                min_y, max_y = min(min_y, y), max(max_y, y)
                min_z, max_z = min(min_z, z), max(max_z, z)

    print(f"Min [{min_x}, {min_y}, {min_z}]")
    print(f"Max [{max_x}, {max_y}, {max_z}]")

# scene_path = "scene/ours/living_room_2/living_room_2.obj"
scene_path = "scene/SceneNetRGBD_Layouts/office/office4_layout.obj"

print("Check min max for", os.path.basename(scene_path)[:-4])
check_min_max(scene_path)
