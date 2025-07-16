import json
from collections.abc import Callable
import numpy
import jax

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        if isinstance(obj, jax.Array):
            return obj.tolist()
        return super().default(obj)
    

def format_dict_inline_lists(data: dict) -> str:
    lines = ["{"]
    for i, (key, val) in enumerate(data.items()):
        key_str = json.dumps(key)  # Properly quoted
        list_str = json.dumps(val, separators=(", ", ": "))  # Inline list

        # Format: key on one line, list on the next
        entry = f'  {key_str}:\n    {list_str}'

        # Add comma if not the last item
        if i < len(data) - 1:
            entry += ","

        lines.append(entry)
    lines.append("}")
    return "\n".join(lines)

def standard_json_formating_dict(dict:dict) -> str:
    return json.dumps(dict, indent=4)

def load_dict_from_JSON(path:str) -> dict:
    try:
        with open(path) as jsonFile:
            dict = json.load(jsonFile)
            return dict
    except:
        raise ValueError

def write_dict_to_JSON(dict: dict, path: str, formating: Callable = standard_json_formating_dict):
    # Convert the dictionary to a JSON string (without indentation for compactness)

    json_File = formating(dict)

    # Save to file
    with open(path, "w") as jsonFile:
        jsonFile.write(json_File)

def load_list_from_JSON(path:str) -> list:
    try:
        with open(path) as jsonFile:
            list = json.load(jsonFile)
            return list
    except:
        raise ValueError

def write_list_to_JSON(list: list, path: str):
    json_File = json.dumps(list, indent=4, cls=NumpyEncoder)

    # Save to file
    with open(path, "w") as jsonFile:
        jsonFile.write(json_File)

def load_set_from_JSON(path:str) -> set:
    retSet = set(load_list_from_JSON(path))
    return retSet

def write_set_to_JSON(set: set, path: str):
    write_list_to_JSON(list(set), path)