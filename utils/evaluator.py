from utils.generator import TestGenerator
from utils.model_ops import evaluate

from keras.optimizers import Adam
from utils.model_ops import age_mae
from keras.models import load_model
import keras
import tensorflow as tf
import random

random.seed(111)

BATCH_SIZE = 1
EVAL_BATCH_SIZE = 32
MODEL_PATH = '/Users/mmatak/dev/thesis/adversarial_framework/model/resnet50-3.436-5.151-sgd.hdf5'
TEST_SET_PATH = '/Users/mmatak/dev/thesis/datasets/appa-real-release'
ADV_SET_PATH = RESULT_PATH = '/Users/mmatak/dev/thesis/datasets/appa-real-release-adv/whitebox/fgsm/'
IMAGE_SIZE = 224


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

# evaluate model on legit dataset
evaluate(model, test_generator, EVAL_BATCH_SIZE)

# load adv dataset
adv_generator = TestGenerator(TEST_SET_PATH, BATCH_SIZE, IMAGE_SIZE)

# evaluate model on adv dataset
evaluate(model, adv_generator, EVAL_BATCH_SIZE)

