import numpy as np
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


def evaluate(sess, x, y, model, generator):
    batch_size = 1

    # load dataset
    x_tmp, y_tmp = zip(*(generator[i] for i in range(len(generator))))
    x_test, y_test = np.vstack(x_tmp), np.vstack(y_tmp)

    print("Dataset loaded")

    # Evaluate the accuracy
    eval_par = {'batch_size': batch_size}

    acc = model_eval(sess, x, y, model.get_logits(x), x_test, y_test, args=eval_par)
    print('Percentage of legit samples that are correctly classified: %0.4f\n' % acc)
