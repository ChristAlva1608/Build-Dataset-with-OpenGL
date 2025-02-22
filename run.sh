#!/bin/bash

# Define arguments
SCENE_PATH="scene/house_interior/house_interior.obj"
OBJ_FOLDER_PATH="obj/GSO"
NUM_IMAGES=10
NUM_OBJECTS=5

# Run Python script with arguments
py main.py \
    --obj_location \
    --obj_rotation \
    --obj_scale \
    --obj_texture \
    --camera_pos \
    --scene_path "$SCENE_PATH" \
    --obj_folder_path "$OBJ_FOLDER_PATH" \
    --num_objects "$NUM_OBJECTS" \
    --num_images "$NUM_IMAGES" \
    --auto