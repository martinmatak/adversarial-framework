import numpy as np


def convert_to_one_hot(target_class, num_of_classes):
    targets_np = np.array([target_class])
    targets_np_one_hot = np.zeros((targets_np.size, num_of_classes))
    targets_np_one_hot[np.arange(targets_np.size), targets_np] = 1
    return targets_np_one_hot
