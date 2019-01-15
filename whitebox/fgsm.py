from utils.generator import TestGenerator
from keras.optimizers import Adam
from keras import metrics
import keras

import tensorflow as tf
import numpy as np
from keras.models import load_model
from cleverhans.utils_keras import KerasModelWrapper
from cleverhans.utils_tf import model_eval
from cleverhans.attacks import FastGradientMethod


BATCH_SIZE = 1
MODEL_PATH = '/Users/mmatak/dev/thesis/adversarial_framework/model/github-pretrained.hdf5'
MODEL_PATH_OVERFIT = '/Users/mmatak/dev/thesis/adversarial_framework/model/weights.044-0.123-1.624.hdf5'
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

model.compile(optimizer=Adam(), loss="categorical_crossentropy", metrics=[metrics.mae, metrics.categorical_accuracy])
wrap = KerasModelWrapper(model)

print("Model loaded")

# load dataset
test_generator = TestGenerator(TEST_SET_PATH, BATCH_SIZE, IMAGE_SIZE)
x_tmp, y_tmp = zip(*(test_generator[i] for i in range(len(test_generator))))
x_test, y_test = np.vstack(x_tmp), np.vstack(y_tmp)

print("Dataset loaded")

# Define input TF placeholder
x = tf.placeholder(tf.float32, shape=(None, IMAGE_SIZE, IMAGE_SIZE, NUM_OF_CHANNELS))
y = tf.placeholder(tf.float32, shape=(None, NB_CLASSES))

# Evaluate the accuracy on legit samples
eval_par = {'batch_size': BATCH_SIZE}
acc = model_eval(sess, x, y, wrap.get_logits(x), x_test, y_test, args=eval_par)
print('Test accuracy on legit examples: %0.4f\n' % acc)

# Initialize the Fast Gradient Sign Method (FGSM) attack object and graph
fgsm = FastGradientMethod(wrap, sess=sess)
fgsm_params = {'eps': 0.3,
               'clip_min': 0.,
               'clip_max': 255.}
adv_x = fgsm.generate(x, **fgsm_params)

# Consider the attack to be constant
adv_x = tf.stop_gradient(adv_x)

# Evaluate the accuracy  on adversarial examples
eval_par = {'batch_size': BATCH_SIZE}
acc = model_eval(sess, x, y, wrap.get_logits(adv_x), x_test, y_test, args=eval_par)
print('Test accuracy on adversarial examples: %0.4f\n' % acc)

