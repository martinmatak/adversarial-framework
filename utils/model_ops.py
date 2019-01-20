import numpy as np
from keras import backend as K


from cleverhans.utils_tf import model_eval


def model_argmax(sess, x, predictions, samples, feed=None):
    """
    Helper function that computes the current class prediction
    :param sess: TF session
    :param x: the input placeholder
    :param predictions: the model's symbolic output
    :param samples: numpy array with input samples (dims must match x)
    :param feed: An optional dictionary that is appended to the feeding
             dictionary before the session runs. Can be used to feed
             the learning phase of a Keras model for instance.
    :return: the argmax output of predictions, i.e. the current predicted class
    """
    feed_dict = {x: samples}
    if feed is not None:
        feed_dict.update(feed)
    probabilities = sess.run(predictions, feed_dict)

    if samples.shape[0] == 1:
        return np.argmax(probabilities)
    else:
        return np.argmax(probabilities, axis=1)


def evaluate(model, generator):
    batch_size = 1

    # load dataset
    x_tmp, y_tmp = zip(*(generator[i] for i in range(len(generator))))
    x_test, y_test = np.vstack(x_tmp), np.vstack(y_tmp)

    print("Dataset loaded")

    result = model.evaluate(x_test, y_test, batch_size, verbose=1)

    print(model.metrics_names[1] + ": " + str(result[1]))

def age_mae(y_true, y_pred):
    true_age = K.sum(y_true * K.arange(0, 101, dtype="float32"), axis=-1)
    pred_age = K.sum(y_pred * K.arange(0, 101, dtype="float32"), axis=-1)
    mae = K.mean(K.abs(true_age - pred_age))
    return mae