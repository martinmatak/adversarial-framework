import tensorflow as tf
import keras
import numpy as np
from utils.generator import TestGenerator
from utils.model_ops import evaluate_generator
from keras.optimizers import Adam
from utils.model_ops import age_mae
from keras.models import load_model
from cleverhans.attacks import FastGradientMethod
from cleverhans.utils_keras import KerasModelWrapper
from whitebox.attacks import fgsm
from utils.image_ops import L2_distance, save_image
from utils.numpy_ops import convert_to_one_hot

BATCH_SIZE = 1
EVAL_BATCH_SIZE = 32
MODEL_PATH = '/Users/mmatak/dev/thesis/adversarial_framework/model/resnet50-3.436-5.151-sgd.hdf5'
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


def train_sub():
    model = load_model(MODEL_PATH, compile=False)

    model.compile(optimizer=Adam(), loss="categorical_crossentropy", metrics=[age_mae])
    wrap = KerasModelWrapper(model)
    print("Model loaded")
    return wrap


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


def blackbox():

    sess = tf.Session()
    keras.backend.set_session(sess)
    print("Session initialized")

    # Seed random number generator so results are reproducible
    rng = np.random.RandomState([2019, 1, 30])

    # simulate the black-box model locally
    print("Preparing the black-box model.")
    target = prep_bbox(sess)

    # train substitute using method from https://arxiv.org/abs/1602.02697
    print("Training the substitute model.")
    substitute = train_sub()

    test_generator = TestGenerator(TEST_SET_PATH, BATCH_SIZE, IMAGE_SIZE)
    # evaluate nets on clean test samples
    evaluate_generator(substitute.model, test_generator, EVAL_BATCH_SIZE)
    evaluate_generator(target.model, test_generator, EVAL_BATCH_SIZE)

    # generate adv samples
    generate_adv_samples(substitute, test_generator, sess)

    # load adv samples
    result_generator = TestGenerator(RESULT_PATH, BATCH_SIZE, IMAGE_SIZE)

    #  Evaluate the accuracy of the substitute model on adversarial examples
    evaluate_generator(substitute.model, result_generator, EVAL_BATCH_SIZE)

    # Evaluate the accuracy of the "black-box" model on adversarial examples
    evaluate_generator(target.model, result_generator, EVAL_BATCH_SIZE)


def main(argv=None):
    blackbox()


if __name__ == '__main__':
    tf.app.run()