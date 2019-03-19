from utils.generator import TestGenerator, TransferGenerator
from utils.image_ops import L2_distance, save_image
from utils.model_ops import evaluate_generator, get_dataset, model_argmax
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
import os

random.seed(111)

# prototype constants
DATASET_PATH = '/Users/mmatak/dev/thesis/datasets/appa-real-release'

# remote constants
# DATASET_PATH = '/root/datasets/appa-real-release/'


TEST_SAMPLES_NAMES = 'resources/100-test-samples.csv'
MODEL_PATH = 'resources/models/resnet50-3.456-6.772-adam.hdf5'

ATTACK_NAME = 'fgsm'
ADV_DATASET_PATH = DATASET_PATH + '-adv/' + 'blackbox/' + ATTACK_NAME + "/"


BATCH_SIZE = 1
EVAL_BATCH_SIZE = 1
IMAGE_SIZE = 224
NUM_OF_CHANNELS = 3
NB_CLASSES = 101


# start tf session
sess = tf.Session()
keras.backend.set_session(sess)

print("Session initialized")

# load model
root_dir = os.path.dirname(os.path.dirname(__file__))
relative_path = os.path.join(root_dir, MODEL_PATH)
model = load_model(relative_path, compile=False)
#model = get_simple_model(NB_CLASSES, IMAGE_SIZE)

model.compile(optimizer=Adam(), loss="categorical_crossentropy", metrics=[age_mae])
#model.compile(optimizer=rmsprop(lr=0.0001, decay=1e-6), loss="categorical_crossentropy", metrics=['accuracy'])

print("Model loaded")

# Define input TF placeholder
x = tf.placeholder(tf.float32, shape=(None, IMAGE_SIZE, IMAGE_SIZE, NUM_OF_CHANNELS))
y = tf.placeholder(tf.float32, shape=(None, NB_CLASSES))

def bbox_predict(wrap, data, sess, x, batch_size=1):
    # here comes API call or anything similar
    print("Num of queries for prediction: " + str(len(data)))
    predictions = model_argmax(sess, x, wrap.get_logits(x), data[:batch_size])
    data = data[batch_size:]
    while len(data) > 0:
        predictions_new = model_argmax(sess, x, wrap.get_logits(x), data[:batch_size])
        predictions = np.hstack([predictions, predictions_new])
        data = data[batch_size:]
    print("predictions: ")
    print(predictions)
    return predictions

print("Loading clean samples...")

# load legit dataset
test_generator = TestGenerator(DATASET_PATH, BATCH_SIZE, IMAGE_SIZE, TEST_SAMPLES_NAMES)

print("Evaluating the accuracy of the model on clean examples...")
data, labels = get_dataset(test_generator)
labels = [np.argmax(label, axis=None, out=None) for label in labels]
labels = [int(label / int(101 / NB_CLASSES)) for label in labels]
clean_generator = TransferGenerator(data, labels, NB_CLASSES, BATCH_SIZE, IMAGE_SIZE)
evaluate_generator(model, clean_generator, EVAL_BATCH_SIZE)


RESULT_PATH = DATASET_PATH + '-adv/whitebox/' + ATTACK_NAME + '/'

wrap = KerasModelWrapper(model)
bbox_predict(wrap, data, sess, x)
if ATTACK_NAME == 'fgsm':
    attack_instance_graph = FastGradientMethod(wrap, sess)
    attack_instance = fgsm
elif ATTACK_NAME == 'cw':
    attack_instance_graph = CarliniWagnerL2(wrap, sess)
    attack_instance = cw
elif ATTACK_NAME == 'jsma':
    attack_instance_graph = SaliencyMapMethod(wrap, sess)
    attack_instance = jsma
else:
    ValueError("Only FGSM, CW (L2) and JSMA attacks are supported")

total_success = 0
evaluated_samples = 0

diff_L2 = []

file_names = test_generator.get_file_names()
image_index = 0

TEN_LABEL = convert_to_one_hot(10, NB_CLASSES)
NINETY_LABEL = convert_to_one_hot(90, NB_CLASSES)
for legit_sample, legit_label in test_generator:

    ground_truth = np.argmax(legit_label)

    if ground_truth > 50:
        adv_x = attack_instance.attack(legit_sample, TEN_LABEL, attack_instance_graph)
    else:
        adv_x = attack_instance.attack(legit_sample, NINETY_LABEL, attack_instance_graph)

    diff_L2.append(L2_distance(legit_sample, adv_x))

    save_image(ADV_DATASET_PATH + 'test/' + file_names[image_index], adv_x[0, :, :, :])
    image_index += 1

print("Average L2 perturbation summed by channels: ", str(sum(diff_L2) / float(len(diff_L2))))
print("Loading adversarial samples...")
result_bbox_generator = TestGenerator(RESULT_PATH, BATCH_SIZE, IMAGE_SIZE, TEST_SAMPLES_NAMES)

print("Evaluating the accuracy of the substitute model on adversarial examples...")
result_data, result_labels = get_dataset(result_bbox_generator)
result_labels = [np.argmax(label, axis=None, out=None) for label in result_labels]
result_labels = [int(label / int(101 / NB_CLASSES)) for label in result_labels]

sub_adv_generator = TransferGenerator(result_data, result_labels, NB_CLASSES, BATCH_SIZE, IMAGE_SIZE)
evaluate_generator(wrap.model, sub_adv_generator, EVAL_BATCH_SIZE)
bbox_predict(wrap, result_data, sess, x)


sess.close()
