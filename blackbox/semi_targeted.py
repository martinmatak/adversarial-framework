import tensorflow as tf
import keras
import numpy as np
from keras.optimizers import Adam
from pathlib import Path
import sys
import os

from six.moves import xrange

from keras.models import load_model

from cleverhans.attacks import FastGradientMethod,  CarliniWagnerL2
from cleverhans.attacks_tf import jacobian_graph, jacobian_augmentation
from cleverhans.utils_keras import KerasModelWrapper


from whitebox.attacks import fgsm, cw
from utils.image_ops import L2_distance, save_image, resize_images
from utils.numpy_ops import convert_to_one_hot, print_statistical_information
from utils.generator import TestGenerator, TransferGenerator
from utils.model_ops import evaluate_generator, age_mae, get_dataset, model_argmax, get_model_category_by_id, get_simple_model, save_model

# prototype constants
DATASET_PATH = '/Users/mmatak/dev/thesis/datasets/appa-real-release'
NUM_EPOCHS = 2

# remote constants
# DATASET_PATH = '/root/datasets/appa-real-release'
# NUM_EPOCHS = 40


TRAINING_SAMPLES_NAMES = 'resources/test-custom-dataset.csv'
TEST_SAMPLES_NAMES = 'resources/test-attack-samples.csv'

bbox_model = sys.argv[1]
print("bbox model: " + bbox_model)
if bbox_model == '1':
    BBOX_MODEL_PATH = 'resources/models/resnet50-3.436-5.151-sgd.hdf5'
    BBOX_IMAGE_SIZE = 224

if bbox_model == '2':
    BBOX_MODEL_PATH = 'resources/models/resnet50-3.456-6.772-adam.hdf5'
    BBOX_IMAGE_SIZE = 224

if bbox_model == '3':
    BBOX_MODEL_PATH = 'resources/models/InceptionResNetV2-3.086-4.505-sgd.hdf5'
    BBOX_IMAGE_SIZE = 299

if bbox_model == '4':
    BBOX_MODEL_PATH = 'resources/models/InceptionResNetV2-3.268-3.922-adam.hdf5'
    BBOX_IMAGE_SIZE = 299

ATTACK_NAME = sys.argv[2]
print("attack: " + ATTACK_NAME)
print("bbox path: " + BBOX_MODEL_PATH)
ADV_DATASET_PATH = DATASET_PATH + '-adv/' + 'blackbox/' + ATTACK_NAME + "/"

SUBSTITUTE_MODEL_ID = sys.argv[3]

SUB_IMAGE_SIZE = 299
if SUBSTITUTE_MODEL_ID != '3' and SUBSTITUTE_MODEL_ID != '4':
    SUB_IMAGE_SIZE = 224


BATCH_SIZE = 2
EVAL_BATCH_SIZE = 2
AUG_BATCH_SIZE = 1
DATA_AUG = 2
NUM_OF_CHANNELS = 3
NB_CLASSES = 101
NB_SUB_CLASSES = 6


def prep_bbox():
    root_dir = os.path.dirname(os.path.dirname(__file__))
    relative_path = os.path.join(root_dir, BBOX_MODEL_PATH)
    model = load_model(relative_path, compile=False)
    model.compile(optimizer=Adam(), loss="categorical_crossentropy", metrics=[age_mae])
    wrap = KerasModelWrapper(model)
    print("Model loaded")
    return wrap


def post_process_predictions(predictions):
    post_process_predictions = np.ones_like(predictions)
    for index, label in enumerate(predictions):
        #print("label: ", label)
        new_label = int(min(label, 99) / int(101/NB_SUB_CLASSES))
        #print("new label: ", new_label)
        post_process_predictions[index] = new_label
    return post_process_predictions


def bbox_predict(model, data, sess, x, batch_size=1):
    # here comes API call or anything similar
    print("querying bbox...")
    data = resize_images(data, BBOX_IMAGE_SIZE)
    predictions = model_argmax(sess, x, model.get_logits(x), data[:batch_size])
    data = data[batch_size:]
    while len(data) > 0:
        predictions_new = model_argmax(sess, x, model.get_logits(x), data[:batch_size])
        predictions = np.hstack([predictions, predictions_new])
        data = data[batch_size:]
    print(predictions)
    post_processed_predictions = post_process_predictions(predictions)
    print(post_processed_predictions)
    print("bbox querying finished")
    print("Num of queries to bbox requsted in this call: " + str(len(post_processed_predictions)))
    return post_processed_predictions


def train_sub(data_aug, sess,
              x_sub, lmbda, target_model, aug_batch_size=AUG_BATCH_SIZE):
    placeholder_sub = tf.placeholder(tf.float32, shape=(None, SUB_IMAGE_SIZE, SUB_IMAGE_SIZE, NUM_OF_CHANNELS))
    placeholder_bbox = tf.placeholder(tf.float32, shape=(None, BBOX_IMAGE_SIZE, BBOX_IMAGE_SIZE, NUM_OF_CHANNELS))

    print("Loading substitute model...")
    model = get_model_category_by_id(SUBSTITUTE_MODEL_ID, NB_SUB_CLASSES)

    # simple vanilla cnn
    if SUBSTITUTE_MODEL_ID == '-1':
        model = get_simple_model(NB_SUB_CLASSES, SUB_IMAGE_SIZE)
        model.compile(optimizer=Adam(lr=0.1, decay=1e-6), loss="categorical_crossentropy", metrics=[age_mae])
    model_sub = KerasModelWrapper(model)

    preds_sub = model_sub.get_logits(placeholder_sub)
    print("Subsitute model loaded.")

    # Define the Jacobian symbolically using TensorFlow
    print("Defining jacobian graph...")
    grads = jacobian_graph(preds_sub, placeholder_sub, NB_SUB_CLASSES)
    print("Jacobian graph defined.")

    y_sub = bbox_predict(target_model, x_sub, sess, placeholder_bbox, batch_size=1)
    train_gen = TransferGenerator(x_sub, labels=y_sub, num_classes=NB_SUB_CLASSES, batch_size=BATCH_SIZE, image_size=SUB_IMAGE_SIZE)
    for rho in xrange(data_aug):
        print("Substitute training epoch #" + str(rho))
        train_gen.reinitialize(x_sub, y_sub, BATCH_SIZE, SUB_IMAGE_SIZE)
        print("Fitting the generator with the labels: ")
        print(train_gen.labels)
        model_sub.model.fit_generator(generator=train_gen, epochs=NUM_EPOCHS)

        # print("Saving substitute model that is trained so far")
        # path = Path(__file__).resolve().parent.parent.joinpath("resources/models")
        # save_model(str(path) + "sub_model_after_epoch" + str(rho) + ".h5", model_sub.model)

        # input_sample = np.empty(shape=(1, IMAGE_SIZE_SUB, IMAGE_SIZE_SUB, NUM_OF_CHANNELS), dtype=np.float32)
        if rho < data_aug - 1:
            print("Augmenting substitute training data...")
            # Perform the Jacobian augmentation
            lmbda_coef = 2 * int(int(rho / 3) != 0) - 1

            x_sub = jacobian_augmentation(sess, placeholder_sub, x_sub, y_sub, grads,
                                          lmbda_coef * lmbda, aug_batch_size)
            print("Substitute training data augmented.")

            print("Labeling substitute training data using bbox...")
            y_sub = np.hstack([y_sub, y_sub])
            x_sub_new = x_sub[int(len(x_sub) / 2):]
            y_sub[int(len(x_sub)/2):] = bbox_predict(target_model, x_sub_new, sess, placeholder_bbox)
    return model_sub


def generate_adv_samples(wrap, generator, sess, file_names=None):
    if ATTACK_NAME == 'fgsm':
        attack_instance_graph = FastGradientMethod(wrap, sess)
        attack_instance = fgsm
    else:
        attack_instance_graph = CarliniWagnerL2(wrap, sess)
        attack_instance = cw

    diff_L2 = []

    if file_names is None:
        file_names = generator.get_file_names()
    image_index = 0

    TEN_LABEL = convert_to_one_hot(10, NB_CLASSES)
    NINETY_LABEL = convert_to_one_hot(90, NB_CLASSES)
    for legit_sample, legit_label in generator:

        ground_truth = np.argmax(legit_label)

        if ground_truth > 50:
            adv_x = attack_instance.attack(legit_sample, None, attack_instance_graph)
        else:
            adv_x = attack_instance.attack(legit_sample, None, attack_instance_graph)

        diff_L2.append(L2_distance(legit_sample, adv_x))

        save_image(ADV_DATASET_PATH + 'test/' + file_names[image_index], adv_x[0, :, :, :])
        image_index += 1

    print("Obtaining statistical information for L2 perturbation summed by channels")
    print_statistical_information(diff_L2)


def assert_equal(L1, L2):
    return len(L1) == len(L2) and sorted(L1) == sorted(L2)


def blackbox(sess):
    # simulate the black-box model locally
    print("Preparing the black-box model.")
    target = prep_bbox()

    # train substitute using method from https://arxiv.org/abs/1602.02697
    print("Training the substitute model by querying the target network..")
    data, _ = get_dataset(TestGenerator(appa_dir=DATASET_PATH,
                                             batch_size=BATCH_SIZE,
                                             image_size=SUB_IMAGE_SIZE,
                                             chosen_samples_path=TRAINING_SAMPLES_NAMES))
    substitute = train_sub(data_aug=DATA_AUG, target_model=target, sess=sess, x_sub=data, lmbda=.1)



    print("Evaluating the accuracy of the substitute model on clean examples...")
    test_data, test_labels = get_dataset(TestGenerator(DATASET_PATH, 1, SUB_IMAGE_SIZE, TEST_SAMPLES_NAMES))
    test_labels = [np.argmax(label, axis=None, out=None) for label in test_labels]
    test_labels = [int(label / int(101/NB_SUB_CLASSES)) for label in test_labels]
    sub_generator = TransferGenerator(test_data, test_labels, NB_SUB_CLASSES, 1, SUB_IMAGE_SIZE)
    evaluate_generator(substitute.model, sub_generator, EVAL_BATCH_SIZE)

    print("Evaluating the accuracy of the black-box model on clean examples...")
    bbox_generator = TestGenerator(DATASET_PATH, 1, SUB_IMAGE_SIZE, TEST_SAMPLES_NAMES)
    evaluate_generator(target.model, bbox_generator, 1)

    print("Generating adversarial samples...")
    generate_adv_samples(substitute, sub_generator, sess, bbox_generator.get_file_names())

    print("Loading adversarial samples...")
    result_bbox_generator = TestGenerator(ADV_DATASET_PATH, BATCH_SIZE, BBOX_IMAGE_SIZE, TEST_SAMPLES_NAMES)

    print("Evaluating the accuracy of the substitute model on adversarial examples...")
    result_data, result_labels = get_dataset(result_bbox_generator)

    result_labels = [np.argmax(label, axis=None, out=None) for label in result_labels]
    result_labels = [int(label / int(101/NB_SUB_CLASSES)) for label in result_labels]

    assert_equal(result_labels, test_labels)
    sub_adv_generator = TransferGenerator(result_data, result_labels, NB_SUB_CLASSES, BATCH_SIZE, SUB_IMAGE_SIZE)
    evaluate_generator(substitute.model, sub_adv_generator, EVAL_BATCH_SIZE)

    print("Evaluating the accuracy of the black-box model on adversarial examples...")
    evaluate_generator(target.model, result_bbox_generator, EVAL_BATCH_SIZE)


def main(argv=None):
    sess = tf.Session()
    keras.backend.set_session(sess)
    print("Session initialized")
    blackbox(sess)


if __name__ == '__main__':
    main()
