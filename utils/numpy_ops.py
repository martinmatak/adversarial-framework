import numpy as np


def convert_to_one_hot(target_class, num_of_classes):
    targets_np = np.array([target_class])
    targets_np_one_hot = np.zeros((targets_np.size, num_of_classes))
    targets_np_one_hot[np.arange(targets_np.size), targets_np] = 1
    return targets_np_one_hot


def print_statistical_information(values):
    print("Computing statistical information")
    print("mean: " + str(compute_mean(values)))
    print("standard deviation: " + str(compute_std_dev(values)))
    print("max value: " + str(get_max_value(values)))
    print("min value: " + str(get_min_value(values)))
    print("-----------------------------------")


def compute_mean(values):
    return np.mean(values)


def compute_std_dev(values):
    return np.std(values)


def get_max_value(values):
    return np.max(values)


def get_min_value(values):
    return np.min(values)