import os
import json

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

def update_label_map(data_dir, file_path):
    """Create/Update label map for data."""
    
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            label_map = json.load(f)
    else:
        label_map = {}
        
    # check sub_dir in data_dir
    current_labels = sorted([
        str(d) for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith('.')
    ])
    
    next_id = max(label_map.values(), default=-1) + 1
    for label in current_labels:
        if label not in label_map:
            label_map[label] = next_id
            next_id += 1

    for lbl in list(label_map):
        if lbl not in current_labels:
            del label_map[lbl]

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)

    return label_map