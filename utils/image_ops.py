import cv2


def save_results(adv_x, attack_type, attack_name, successful, original_class, target_class):
    image_name = 'adv_img-'+ attack_type + '-'  + attack_name + '-' + \
                 str(original_class) + '-' + str(target_class) + '-' + str(successful) + '.jpg'
    directory = '../results'
    path = directory + '/' + image_name
    cv2.imwrite(path, adv_x)
