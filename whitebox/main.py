from utils.generator import TestGenerator
from keras.optimizers import Adam
from keras import metrics
from utils.image_ops import save_results
from utils.model_ops import model_argmax, evaluate
from utils.numpy_ops import convert_to_one_hot
import keras

import tensorflow as tf
import numpy as np
from keras.models import load_model
from cleverhans.utils_keras import KerasModelWrapper
from cleverhans.attacks import FastGradientMethod, CarliniWagnerL2, SaliencyMapMethod
from whitebox.attacks import fgsm, cw, jsma


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

# load legit dataset
test_generator = TestGenerator(TEST_SET_PATH, BATCH_SIZE, IMAGE_SIZE)

# Define input TF placeholder
x = tf.placeholder(tf.float32, shape=(None, IMAGE_SIZE, IMAGE_SIZE, NUM_OF_CHANNELS))
y = tf.placeholder(tf.float32, shape=(None, NB_CLASSES))

# evaluate model on legit dataset
evaluate(sess, x, y, wrap, test_generator)

# pick the attack

attack = 'fgsm'
#attack = 'cw'

# not working because of memory consumption
#attack = 'jsma'

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

target_class = 88
target_class_encoded = convert_to_one_hot(target_class, NB_CLASSES)
print("Generating adv. samples for target class %i" % target_class)

for legit_sample, legit_label in test_generator:

    ground_truth = np.argmax(legit_label)
    print("[Original sample] ground truth: ", ground_truth)

    predicted_class = int(model_argmax(sess, x, wrap.get_logits(x), legit_sample))
    print("[Original sample] predicted class: ", predicted_class)

    if ground_truth != predicted_class:
        print("Skipping sample because not correctly predicted")
        print(" ")
        continue
    else:
        evaluated_samples += 1

    adv_x = attack_instance.attack(legit_sample, target_class_encoded, attack_instance_graph)

    predicted_class = int(model_argmax(sess, x, wrap.get_logits(x), adv_x))
    print("[Adversarial sample] predicted class: ", predicted_class)

    save_results(adv_x[0, :, :, :], 'whitebox', 'fgsm', predicted_class == target_class, ground_truth, target_class)

    if predicted_class == target_class:
        print("Attack was successful")
        total_success += 1
    else:
        print("Attack was not successful")

    print(" ")

print("Total # of successful targeted attacks: " + str(total_success) + " / " + str(evaluated_samples))

sess.close()