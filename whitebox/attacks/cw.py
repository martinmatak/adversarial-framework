def attack(sample, target_np_one_hot, cw):

    cw_params = {'binary_search_steps': 8,
                 'y_target': target_np_one_hot,
                 'abort_early': True,
                 'max_iterations': 5000,
                 'learning_rate': 1,
                 'clip_max': 255,
                 'clip_min': 0,
                 'initial_const': 0.1}
    adv_x = cw.generate_np(sample, **cw_params)
    return adv_x
