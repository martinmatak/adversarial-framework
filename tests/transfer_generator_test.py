from utils.generator import TestGenerator, TransferGenerator
from utils.model_ops import evaluate_generator, get_dataset

from keras.optimizers import Adam
from utils.model_ops import age_mae
from keras.models import load_model
import keras
import tensorflow as tf
import random
import numpy as np

random.seed(111)

BATCH_SIZE = 1
EVAL_BATCH_SIZE = 32
MODEL_PATH = '/Users/mmatak/dev/thesis/adversarial_framework/model/resnet50-3.436-5.151-sgd.hdf5'
TEST_SET_PATH = '/Users/mmatak/dev/thesis/datasets/appa-real-release-overfit'
IMAGE_SIZE = 224
NUM_OF_CHANNELS = 3
NB_CLASSES = 101


# start tf session
sess = tf.Session()
keras.backend.set_session(sess)

print("Session initialized")

# load model
model = load_model(MODEL_PATH, compile=False)

model.compile(optimizer=Adam(), loss="categorical_crossentropy", metrics=[age_mae])

print("Model loaded")

# load legit dataset
test_generator = TestGenerator(TEST_SET_PATH, BATCH_SIZE, IMAGE_SIZE)

#evaluate_generator(model, test_generator, EVAL_BATCH_SIZE)

x_data, y_data = get_dataset(test_generator)

y_classes = [np.argmax(y, axis=None, out=None) for y in y_data]

new_generator = TransferGenerator(x_data, y_classes, BATCH_SIZE, IMAGE_SIZE)

#evaluate_generator(model, new_generator, EVAL_BATCH_SIZE)

x_data_new, y_data_new = get_dataset(new_generator)

assert np.array_equal(x_data, x_data_new)
assert np.array_equal(y_data, y_data_new)
print("Test for transfer generator successfull!")
sess.close()
