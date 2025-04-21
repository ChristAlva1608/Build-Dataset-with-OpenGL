import yaml
from PIL import Image
import numpy as np
import json
from labels import Label
import os

LABELS_FILE = "instance_labels.json"

def read_yaml_file(file_path):
    try:
        with open(file_path, 'r') as file:
            content = yaml.safe_load(file)
            return content
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except yaml.YAMLError as e:
        print(f"Error: An error occurred while parsing the YAML file.\n{e}")
    return None


def valid_depth_map(depth_image):
    mask_depth = depth_image[depth_image > 0]
    std = np.std(mask_depth)

    # mask percentage
    zero_mask = (depth_image == 0)
    masked_percentage = np.mean(zero_mask) * 100

    # close pixels percetage
    close_mask = (depth_image < 1000)
    close_percentage = np.mean(close_mask) * 100

    if (std < 500 or masked_percentage > 10 or close_percentage > 20): # is not valid
        # print("Standard deviation", np.std(mask_depth))
        # print("Mask percentage", masked_percentage)
        # print("Close percentage", close_percentage)
        return False
    return True


def get_all_NYU_rgb_images(dataset_path):
    image_paths = []
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.startswith("rgb_") and file.lower().endswith((".jpg")):
                image_paths.append(os.path.join(root, file))
    return image_paths


def load_labels():
    if os.path.exists(LABELS_FILE):
        with open(LABELS_FILE, 'r') as f:
            raw = json.load(f)
        return {k: Label(k, v['id'], v['color']) for k, v in raw.items()}
    else:
        return {}

def save_labels(labels):
    data = {name: {'id': label.id, 'color': label.color} for name, label in labels.items()}
    with open(LABELS_FILE, 'w') as f:
        json.dump(data, f, indent=2)