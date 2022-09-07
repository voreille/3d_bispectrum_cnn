import json

def get_split(split_id, split_file):
    with open(split_file) as f:
        meta = json.load(f)
    return meta[split_id]