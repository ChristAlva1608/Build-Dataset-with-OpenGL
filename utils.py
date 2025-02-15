import yaml

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
