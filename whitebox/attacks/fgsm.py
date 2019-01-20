def attack(sample, target_np_one_hot, fgsm):

    fgsm_params = {'eps': 5,
                   'clip_min': 0.,
                   'clip_max': 255.,
                   'y_target': target_np_one_hot,
                   }
    adv_x = fgsm.generate_np(sample, **fgsm_params)
    return adv_x
