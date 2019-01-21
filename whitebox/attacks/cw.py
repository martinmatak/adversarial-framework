def attack(sample, target_np_one_hot, cw):

    cw_params = {'binary_search_steps': 4,
                 'y_target': target_np_one_hot,
                 'abort_early': True,
                 'max_iterations': 1000,
                 'learning_rate': 0.1,
                 'clip_max': 255.,
                 'clip_min': 0.,
                 'initial_const': 10}
    adv_x = cw.generate_np(sample, **cw_params)
    return adv_x
