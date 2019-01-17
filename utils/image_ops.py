import cv2
import numpy as np


def save_results(adv_x, attack_type, attack_name, successful, original_class, target_class):
    image_name = 'adv_img-'+ attack_type + '-'  + attack_name + '-' + \
                 str(original_class) + '-' + str(target_class) + '-' + str(successful) + '.jpg'
    directory = '../results'
    path = directory + '/' + image_name
    cv2.imwrite(path, adv_x)


# def L0_distance(image_a, image_b):
#     return L_distance(image_a, image_b, 0)
#
#
# def L1_distance(image_a, image_b):
#     return L_distance(image_a, image_b, 1)
#

def L2_distance(image_a, image_b):
    return np.sum((image_a-image_b)**2)**.5


# def Linf_distance(image_a, image_b):
#     return L_distance(image_a, image_b, np.inf)
#
#
# def L_distance(image_a, image_b, order):
#     diff = image_a - image_b
#     return np.linalg.norm(diff, ord=order, axis=1) \
#            + np.linalg.norm(diff, ord=order, axis=2) \
#            + np.linalg.norm(diff, ord=order, axis=3)
