from utils.generator import TestGenerator
from utils.image_ops import L2_distance, save_image
from utils.model_ops import evaluate_generator
from utils.numpy_ops import convert_to_one_hot
from whitebox.attacks import fgsm, cw, jsma

from keras.optimizers import Adam
from utils.model_ops import age_mae
from keras.models import load_model
from cleverhans.utils_keras import KerasModelWrapper
from cleverhans.attacks import FastGradientMethod, CarliniWagnerL2, SaliencyMapMethod
import keras
import tensorflow as tf
import numpy as np
import random

attack = 'cw'

random.seed(111)

BATCH_SIZE = 1
EVAL_BATCH_SIZE = 32
MODEL_PATH = '/Users/mmatak/dev/thesis/adversarial_framework/model/github-pretrained.hdf5'
TEST_SET_PATH = '/Users/mmatak/dev/thesis/datasets/appa-real-release-100'
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

# Define input TF placeholder
x = tf.placeholder(tf.float32, shape=(None, IMAGE_SIZE, IMAGE_SIZE, NUM_OF_CHANNELS))
y = tf.placeholder(tf.float32, shape=(None, NB_CLASSES))

# evaluate model
# evaluate(model, test_generator, EVAL_BATCH_SIZE)

# pick the attack
#attack = 'fgsm'
#attack = 'cw'

# not working because of memory consumption
#attack = 'jsma'

RESULT_PATH = '/Users/mmatak/dev/thesis/datasets/appa-real-release-adv/whitebox/' + attack

wrap = KerasModelWrapper(model)
if attack == 'fgsm':
    attack_instance_graph = FastGradientMethod(wrap, sess)
    attack_instance = fgsm
elif attack == 'cw':
    attack_instance_graph = CarliniWagnerL2(wrap, sess)
    attack_instance = cw
elif attack == 'jsma':
    attack_instance_graph = SaliencyMapMethod(wrap, sess)
    attack_instance = jsma
else:
    ValueError("Only FGSM, CW (L2) and JSMA attacks are supported")

total_success = 0
evaluated_samples = 0

diff_L2 = []

img_ids = [str("00" + str(i)) for i in range(5613, 7613)]
id_index = 0

TEN_LABEL = convert_to_one_hot(10, NB_CLASSES)
NINETY_LABEL = convert_to_one_hot(90, NB_CLASSES)
for legit_sample, legit_label in test_generator:

    ground_truth = np.argmax(legit_label)

    if ground_truth > 50:
        adv_x = attack_instance.attack(legit_sample, TEN_LABEL, attack_instance_graph)
    else:
        adv_x = attack_instance.attack(legit_sample, NINETY_LABEL, attack_instance_graph)

    diff_L2.append(L2_distance(legit_sample, adv_x))

    save_image(RESULT_PATH + '/test/' + img_ids[id_index] + ".jpg_face.jpg", adv_x[0, :, :, :])
    id_index += 1

print("Average L2 perturbation summed by channels: ", str(sum(diff_L2) / float(len(diff_L2))))

#evaluate on a new dataset
result_generator = TestGenerator(RESULT_PATH, BATCH_SIZE, IMAGE_SIZE)

evaluate_generator(wrap.model, result_generator, EVAL_BATCH_SIZE)

sess.close()
