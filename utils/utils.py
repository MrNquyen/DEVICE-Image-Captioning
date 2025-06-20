import json
import numpy as np
import yaml

#---- Load json
def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        json_content = json.load(file)
        return json_content
    
#---- Save json
def save_json(path, content):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(content, file, ensure_ascii=False, indent=3)


#---- Load numpy
def load_npy(path):
    return np.load(path, allow_pickle=True)


#---- Load vocab
def load_vocab(path):
    with open(path, "r", encoding="utf-8") as f:
        vocab = [line.strip() for line in f]
        return vocab

#---- Get name of the image
def get_img_name(name):
    return name.split(".")[0]

#---- Load yaml file
def load_yml(path):
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
        return config