import yaml
from PIL import Image
import numpy as np

def read_yaml_file(file_path):
    """
    Reads a YAML file and returns its content.

    Args:
        file_path (str): The path to the YAML file.

    Returns:
        dict: Parsed content of the YAML file.
    """
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

    # close mask percetage
    close_mask = (depth_image < 500)
    close_percentage = np.mean(close_mask) * 100

    if (std < 300 or masked_percentage > 20 or close_percentage > 20): # is not valid
        # print("Standard deviation", np.std(mask_depth))
        # print("Mask percentage", masked_percentage)
        # print("Close percentage", close_percentage)
        return False
    return True