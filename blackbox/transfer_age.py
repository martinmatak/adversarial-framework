import tensorflow as tf
import keras
import numpy as np
from utils.generator import TestGenerator
from utils.model_ops import evaluate_generator
from cleverhans.attacks import FastGradientMethod
from whitebox.attacks import fgsm

BATCH_SIZE = 1
EVAL_BATCH_SIZE = 32
MODEL_PATH = '/home/lv71235/mmatak/adversarial-framework/models/resnet50-3.436-5.151-sgd.hdf5'
TEST_SET_PATH = '/home/lv71235/mmatak/datasets/appa-real-release'
RESULT_PATH = TEST_SET_PATH + '-adv/blackbox/'
IMAGE_SIZE = 224
NUM_OF_CHANNELS = 3
NB_CLASSES = 101


def prep_bbox(sess):
    # load model
    # return model, model.get_logits()
    return 1, 2


def train_sub():
    pass


def generate_adv_samples(substitute):
    pass


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

    # evaluate substitute on clean test samples
    test_generator = TestGenerator(TEST_SET_PATH, BATCH_SIZE, IMAGE_SIZE)
    evaluate_generator(substitute.model, test_generator, EVAL_BATCH_SIZE)

    # generate adv samples
    generate_adv_samples(substitute)

    # load adv samples
    result_generator = TestGenerator(RESULT_PATH, BATCH_SIZE, IMAGE_SIZE)

    #  Evaluate the accuracy of the substitute model on adversarial examples
    evaluate_generator(substitute.model, result_generator, EVAL_BATCH_SIZE)

    # Evaluate the accuracy of the "black-box" model on adversarial examples
    evaluate_generator(target.model, result_generator, EVAL_BATCH_SIZE)
