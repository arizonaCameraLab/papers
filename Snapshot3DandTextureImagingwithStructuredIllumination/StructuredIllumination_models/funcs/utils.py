import json
import numpy as np

def json_loader(file_path: str) -> dict:
    """
    Load calibration data from a JSON file.
    Convert each value to a NumPy array or scalar if it contains only one element.
    """
    with open(file_path, "r") as f:
        data = json.load(f)

    result = {}
    for key, value in data.items():
        arr = np.array(value)
        # if array has only one element, return scalar
        result[key] = arr.item() if arr.size == 1 else arr

    return result
