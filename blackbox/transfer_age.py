import tensorflow as tf
import keras
import numpy as np
from keras.optimizers import Adam

from six.moves import xrange

from keras.models import load_model

from cleverhans.attacks import FastGradientMethod
from cleverhans.attacks_tf import jacobian_graph, jacobian_augmentation
from cleverhans.utils_keras import KerasModelWrapper


from whitebox.attacks import fgsm
from utils.image_ops import L2_distance, save_image
from utils.numpy_ops import convert_to_one_hot
from utils.generator import TestGenerator
from utils.model_ops import evaluate_generator, train_model, age_mae, get_dataset

BATCH_SIZE = 1
EVAL_BATCH_SIZE = 32
MODEL_PATH = '/Users/mmatak/dev/thesis/adversarial_framework/model/resnet50-3.436-5.151-sgd.hdf5'
FRESH_MODEL_PATH = '/Users/mmatak/dev/thesis/adversarial_framework/model/resnet50-3.456-6.772-adam.hdf5'
TEST_SET_PATH = '/Users/mmatak/dev/thesis/datasets/appa-real-release-100'
RESULT_PATH = TEST_SET_PATH + '-adv/blackbox/'
IMAGE_SIZE = 224
NUM_OF_CHANNELS = 3
NB_CLASSES = 101


def prep_bbox(sess):
    # load model
    # return model, model.get_logits()
    # load model
    model = load_model(MODEL_PATH, compile=False)

    model.compile(optimizer=Adam(), loss="categorical_crossentropy", metrics=[age_mae])
    wrap = KerasModelWrapper(model)
    print("Model loaded")
    return wrap


def bbox_predict(model, data):
    #TODO
    return [1]


def train_sub(data_aug, nb_epochs_s, batch_size, learning_rate, sess,
              x_sub, y_sub, lmbda, rng, target_model, aug_batch_size=1):
    """

    :param data_aug:
    :param nb_epochs_s:
    :param batch_size:
    :param learning_rate:
    :param sess:
    :param x_sub: initial substitute training data
    :param y_sub: initial substitute training labels in categorical representation
    :param rng: numpy.random.RandomState instance
    :return:
    """
    x = tf.placeholder(tf.float32, shape=(None, IMAGE_SIZE, IMAGE_SIZE,
                                          NUM_OF_CHANNELS))

    #TODO: CREATE fresh model instead, but initialize all variables
    model = load_model(MODEL_PATH, compile=False)
    model.compile(optimizer=Adam(), loss="categorical_crossentropy", metrics=[age_mae])
    model_sub = KerasModelWrapper(model)

    preds_sub = model_sub.get_logits(x)
    print("Defined TensorFlow model graph for the substitute.")

    # Define the Jacobian symbolically using TensorFlow
    grads = jacobian_graph(preds_sub, x, NB_CLASSES)
    print("Defined jacobian graph.")

    for rho in xrange(data_aug):
        print("Substitute training epoch #" + str(rho))
        train_model(model_sub.model, sess, x_sub, y_sub, rng)
        if rho < data_aug - 1:

            print("Augmenting substitute training data")
            # Perform the Jacobian augmentation
            lmbda_coef = 2 * int(int(rho / 3) != 0) - 1
            x_sub = jacobian_augmentation(
                sess=sess,
                x=x,
                X_sub_prev=x_sub,
                Y_sub=y_sub,
                grads=grads,
                lmbda=lmbda_coef*lmbda,
                aug_batch_size=aug_batch_size
            )

            print("Labeling substitute training data using bbox.")
            y_sub = np.hstack([y_sub, y_sub])
            x_sub_prev = x_sub[int(len(x_sub) / 2):]
            predictions = bbox_predict(target_model, x_sub_prev)
            y_sub[int(len(x_sub)/2):] = predictions
    return model_sub


def generate_adv_samples(wrap, generator, sess):
    attack_instance_graph = FastGradientMethod(wrap, sess)
    attack_instance = fgsm

    diff_L2 = []

    img_ids = [str("00" + str(i)) for i in range(5613, 7613)]
    id_index = 0

    TEN_LABEL = convert_to_one_hot(10, NB_CLASSES)
    NINETY_LABEL = convert_to_one_hot(90, NB_CLASSES)
    for legit_sample, legit_label in generator:

        ground_truth = np.argmax(legit_label)

        if ground_truth > 50:
            adv_x = attack_instance.attack(legit_sample, TEN_LABEL, attack_instance_graph)
        else:
            adv_x = attack_instance.attack(legit_sample, NINETY_LABEL, attack_instance_graph)

        diff_L2.append(L2_distance(legit_sample, adv_x))

        save_image(RESULT_PATH + '/test/' + img_ids[id_index] + ".jpg_face.jpg", adv_x[0, :, :, :])
        id_index += 1

    print("Average L2 perturbation summed by channels: ", str(sum(diff_L2) / float(len(diff_L2))))


def blackbox(sess):

    # Seed random number generator so results are reproducible
    rng = np.random.RandomState([2019, 1, 30])

    # simulate the black-box model locally
    print("Preparing the black-box model.")
    target = prep_bbox(sess)

    test_generator = TestGenerator(TEST_SET_PATH, BATCH_SIZE, IMAGE_SIZE)
    # train substitute using method from https://arxiv.org/abs/1602.02697
    print("Training the substitute model.")
    data, labels = get_dataset(test_generator)
    labels = [np.argmax(label) for label in labels]
    substitute = train_sub(data_aug=2, nb_epochs_s=1, batch_size=1,
                           learning_rate=0.01, sess=sess,
                           x_sub=data, y_sub=labels, rng=rng,
                           target_model=target, aug_batch_size=1, lmbda=.1)

    print("Evaluating the accuracy of the substitute model on clean examples")
    evaluate_generator(substitute.model, test_generator, EVAL_BATCH_SIZE)

    print("Evaluating the accuracy of the black-box model on clean examples")
    evaluate_generator(target.model, test_generator, EVAL_BATCH_SIZE)

    print("Generating adversarial samples")
    generate_adv_samples(substitute, test_generator, sess)

    print("Loading adversarial samples")
    result_generator = TestGenerator(RESULT_PATH, BATCH_SIZE, IMAGE_SIZE)

    print("Evaluating the accuracy of the substitute model on adversarial examples")
    evaluate_generator(substitute.model, result_generator, EVAL_BATCH_SIZE)

    print("Evaluating the accuracy of the black-box model on adversarial examples")
    evaluate_generator(target.model, result_generator, EVAL_BATCH_SIZE)


def main(argv=None):
    sess = tf.Session()
    #sess.run(tf.global_variables_initializer())
    keras.backend.set_session(sess)
    print("Session initialized")
    blackbox(sess)


if __name__ == '__main__':
    main()