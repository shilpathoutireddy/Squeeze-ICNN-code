import pickle
from os import path


def load(x):
    with open(x + '.pkl', 'rb') as f:  # 'rb' for reading; can be omitted
        my_dict = pickle.load(f)  # load file content as my_dict
    return my_dict


def save(file, name, override=False):
    if not path.exists(name + '.pkl'):
        with open(name + '.pkl', 'wb') as f:
            pickle.dump(file, f)

    elif override:
        with open(name + '.pkl', 'wb') as f:
            pickle.dump(file, f)
    else:
        pass
