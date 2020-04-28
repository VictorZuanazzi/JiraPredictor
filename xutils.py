import json
from pickle import load, dump

def read_json(path):
    """opens a json file given the path.
    :arg
        path: str, path to folder + file name +.json
    :return
        list of dicts from the decoded json file."""

    try:
        with open(path) as json_file:
            data_dict = json.load(json_file)

        data_list = [data_dict]
    except:
        data_list = []
        for line in open(path, 'r'):
            data_list.append(json.loads(line))

    return data_list

def save_pickle(data_obj, path):
    """helper function to save data
    :arg
        data_obj: any object, object to be saved;
        path: str, relative or full path to save the object."""
    with open(path, 'wb') as handle:
        dump(data_obj, handle)

def load_pickle(path):
    """helper function to load data
    :arg
        path: str, relative or full path to load the object.
    :return
        object saved in path."""
    with open(path, 'rb') as handle:
        data = load(handle)

    return data