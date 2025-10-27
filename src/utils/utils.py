import os

def create_directory(root_dir, sub_dir):
    path = os.path.join(root_dir, sub_dir)
    try:
        os.makedirs(path)
    except OSError:
        print(f"Directory {sub_dir} already exists!")
    except Exception as e:
        print(f"error create {path}: {e}")

def list_files(dir_path):
    """Listing name of all files from directory path."""
    
    results = []
    try:
        for file_name in os.listdir(dir_path):
            if os.path.isfile(os.path.join(dir_path, file_name)):
                results.append(file_name)
    except FileNotFoundError:
        print(f"the directory {dir_path} is not exist.")
    except PermissionError:
        print(f"permission denied to access the directory {dir_path}")
    except OSError as e:
        print(f"An OS error occurred: {e}")
    
    return results