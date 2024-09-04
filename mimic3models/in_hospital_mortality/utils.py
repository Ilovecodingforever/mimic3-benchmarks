from mimic3models import common_utils
import numpy as np
import os


def load_data(reader, discretizer, normalizer, small_part=False, return_names=False):




    # small_part = True





    N = reader.get_number_of_examples()
    if small_part:
        N = 1000
        # N = 4000   # 14681 in total
    ret = common_utils.read_chunk(reader, N)
    data = ret["X"]
    ts = ret["t"]
    labels = ret["y"]
    names = ret["name"]
    data = [discretizer.transform(X, end=t)[0] for (X, t) in zip(data, ts)]
    if normalizer is not None:
        data = [normalizer.transform(X) for X in data]

    if discretizer._same_length:
        whole_data = (np.array(data), labels)
    else:
        whole_data = (data, labels)
    if not return_names:
        return whole_data
    return {"data": whole_data, "names": names}


def save_results(names, pred, y_true, path):
    common_utils.create_directory(os.path.dirname(path))
    with open(path, 'w') as f:
        f.write("stay,prediction,y_true\n")
        for (name, x, y) in zip(names, pred, y_true):
            f.write("{},{:.6f},{}\n".format(name, x, y))
