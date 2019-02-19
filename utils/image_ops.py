import cv2
import numpy as np


def save_results(adv_x, attack_type, attack_name, successful, original_class, target_class):
    image_name = 'adv_img-'+ attack_type + '-'  + attack_name + '-' + \
                 str(original_class) + '-' + str(target_class) + '-' + str(successful) + '.jpg'
    directory = '../results'
    path = directory + '/' + image_name
    save_image(path, adv_x)

def save_image(path, image):
    cv2.imwrite(path, image)


def load_image(image_path, image_size):
    print("Loading ", image_path)
    image = cv2.imread(str(image_path))
    return cv2.resize(image, (image_size, image_size))


def L2_distance(image_a, image_b):
    return np.sum((image_a-image_b)**2)**.5


def resize_images(images, dimension):
    resized_images = np.zeros((len(images), dimension, dimension, 3), dtype=np.uint8)
    for index, sample in enumerate(images):
        resized_images[index] = cv2.resize(sample, (dimension, dimension))
    return resized_images