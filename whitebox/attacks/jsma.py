def attack(sample, target_np_one_hot, jsma):

    jsma_params = {'clip_min': 0.,
                   'clip_max': 255.,
                   'y_target': target_np_one_hot
                   }
    adv_x = jsma.generate_np(sample, **jsma_params)
    return adv_x
