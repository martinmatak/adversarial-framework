import tensorflow as tf
import keras
import numpy as np
import os

from keras.optimizers import Adam, SGD
from six.moves import xrange

from keras.models import load_model
from client_api import FaceAPI

from cleverhans.attacks import FastGradientMethod,  CarliniWagnerL2
from cleverhans.attacks_tf import jacobian_graph, jacobian_augmentation
from cleverhans.utils_keras import KerasModelWrapper


from whitebox.attacks import fgsm, cw
from utils.image_ops import L2_distance, save_image
from utils.numpy_ops import convert_to_one_hot
from utils.generator import TestGenerator, TransferGenerator
from utils.model_ops import evaluate_generator, age_mae, get_dataset, model_argmax, get_model, get_simple_model

# prototype constants
DATASET_PATH = '/Users/mmatak/dev/thesis/datasets/appa-real-release'
NUM_EPOCHS = 1

# remote constants
# DATASET_PATH = '/root/datasets/appa-real-release'
# NUM_EPOCHS = 40


TRAINING_SAMPLES_NAMES = 'resources/test-custom-dataset.csv'
TEST_SAMPLES_NAMES = 'resources/test-attack-samples.csv'
BBOX_MODEL_PATH = 'resources/models/resnet50-3.436-5.151-sgd.hdf5'

ATTACK_NAME = 'fgsm'
ADV_DATASET_PATH = DATASET_PATH + '-adv/' + 'blackbox/' + ATTACK_NAME + "/"


BATCH_SIZE = 1
EVAL_BATCH_SIZE = 1
IMAGE_SIZE = 224
NUM_OF_CHANNELS = 3
NB_CLASSES = 101



def prep_bbox():
    root_dir = os.path.dirname(os.path.dirname(__file__))
    relative_path = os.path.join(root_dir, BBOX_MODEL_PATH)
    model = load_model(relative_path, compile=False)
    model.compile(optimizer=Adam(), loss="categorical_crossentropy", metrics=[age_mae])
    wrap = KerasModelWrapper(model)
    print("Model loaded")
    return wrap


def predict_external_api(np_image):
    return np.array(FaceAPI.predict_from_numpy(np_image))


def bbox_predict(model, data, sess, x, batch_size=1, faceAPI = False):
    # here comes API call or anything similar
    print("Num of queries to bbox: " + str(len(data)))
    if faceAPI:
        predictions = predict_external_api(data[:1][0, :, :, :])
    else:
        predictions = model_argmax(sess, x, model.get_logits(x), data[:batch_size])
    data = data[batch_size:]
    while len(data) > 0:
        if faceAPI:
            predictions_new = predict_external_api(data[:1][0, :, :, :])
        else:
            predictions_new = model_argmax(sess, x, model.get_logits(x), data[:batch_size])
        predictions = np.hstack([predictions, predictions_new])
        data = data[batch_size:]
    print(predictions)
    return predictions


def train_sub(data_aug, sess,
              x_sub, y_sub, lmbda, target_model, aug_batch_size=1):
    x = tf.placeholder(tf.float32, shape=(None, IMAGE_SIZE, IMAGE_SIZE,
                                          NUM_OF_CHANNELS))
    print("Loading substitute model...")
    model = get_model("InceptionResNetV2")
    # model = get_simple_model(num_classes=NB_CLASSES, image_size=IMAGE_SIZE)
    model.compile(optimizer=Adam(), loss="categorical_crossentropy", metrics=[age_mae])
    model_sub = KerasModelWrapper(model)

    preds_sub = model_sub.get_logits(x)
    print("Subsitute model loaded.")

    # Define the Jacobian symbolically using TensorFlow
    print("Defining jacobian graph...")
    grads = jacobian_graph(preds_sub, x, NB_CLASSES)
    print("Jacobian graph defined.")

    train_gen = TransferGenerator(x_sub, y_sub, num_classes=101, batch_size=BATCH_SIZE, image_size=IMAGE_SIZE)
    for rho in xrange(data_aug):
        print("Substitute training epoch #" + str(rho))
        train_gen.reinitialize(data=x_sub, labels=y_sub, batch_size=BATCH_SIZE, image_size=IMAGE_SIZE)
        model_sub.model.fit_generator(generator=train_gen, epochs=1)

        input_sample = np.empty(shape=(1, IMAGE_SIZE, IMAGE_SIZE, NUM_OF_CHANNELS), dtype=np.float32)
        if rho < data_aug - 1:
            print("Augmenting substitute training data...")
            # Perform the Jacobian augmentation
            lmbda_coef = 2 * int(int(rho / 3) != 0) - 1

            x_sub_tmp = np.vstack([x_sub, x_sub])
            for i in range(0, len(y_sub)):
                input_sample[0, :, :, :] = x_sub[i]
                adv = jacobian_augmentation(
                    sess=sess,
                    x=x,
                    X_sub_prev=input_sample,
                    Y_sub=[y_sub[i]],
                    grads=grads,
                    lmbda=lmbda_coef*lmbda,
                    aug_batch_size=aug_batch_size
                )
                x_sub_tmp[2*i] = adv[0, :, :, :]
                x_sub_tmp[2*i + 1] = adv[1, :, :, :]

            x_sub = x_sub_tmp
            print("Substitute training data augmented.")

            print("Labeling substitute training data using bbox...")
            y_sub = np.hstack([y_sub, y_sub])
            x_sub_prev = x_sub[int(len(x_sub) / 2):]
            predictions = bbox_predict(target_model, x_sub_prev, sess, x)
            y_sub[int(len(x_sub)/2):] = predictions
    return model_sub


def train_sub_no_augmn(data, target_model, sess):
    print("Loading a substitute model...")

    x = tf.placeholder(tf.float32, shape=(None, IMAGE_SIZE, IMAGE_SIZE, NUM_OF_CHANNELS))

    model = get_model("ResNet50", NB_CLASSES)
    # model = get_simple_model(num_classes=NB_CLASSES, image_size=IMAGE_SIZE)
    model.compile(optimizer=SGD(), loss="categorical_crossentropy", metrics=[age_mae])
    model_sub = KerasModelWrapper(model)

    print("Substitute model loaded")

    print("Labeling samples...")
    labels = bbox_predict(target_model, data, sess, x, batch_size=1)
    print("Samples labeled")

    print("Training a substitute model...")
    train_gen = TransferGenerator(data, labels, BATCH_SIZE, IMAGE_SIZE)
    model_sub.model.fit_generator(generator=train_gen, epochs=NUM_EPOCHS, verbose=0)
    print("Subsitute model trained")

    return model_sub


def generate_adv_samples(wrap, generator, sess):
    if ATTACK_NAME == 'fgsm':
        attack_instance_graph = FastGradientMethod(wrap, sess)
        attack_instance = fgsm
    else:
        attack_instance_graph = CarliniWagnerL2(wrap, sess)
        attack_instance = cw

    diff_L2 = []

    file_names = generator.get_file_names()
    image_index = 0

    TEN_LABEL = convert_to_one_hot(10, NB_CLASSES)
    NINETY_LABEL = convert_to_one_hot(90, NB_CLASSES)
    for legit_sample, legit_label in generator:

        ground_truth = np.argmax(legit_label)

        if ground_truth > 50:
            adv_x = attack_instance.attack(legit_sample, TEN_LABEL, attack_instance_graph)
        else:
            adv_x = attack_instance.attack(legit_sample, NINETY_LABEL, attack_instance_graph)

        diff_L2.append(L2_distance(legit_sample, adv_x))

        save_image(ADV_DATASET_PATH + 'test/' + file_names[image_index], adv_x[0, :, :, :])
        image_index += 1

    print("Average L2 perturbation summed by channels: ", str(sum(diff_L2) / float(len(diff_L2))))


def blackbox(sess):
    # simulate the black-box model locally
    print("Preparing the black-box model.")
    target = prep_bbox()

    # train substitute using method from https://arxiv.org/abs/1602.02697
    print("Training the substitute model by querying the target network..")
    data, labels = get_dataset(TestGenerator(DATASET_PATH, BATCH_SIZE, IMAGE_SIZE, TRAINING_SAMPLES_NAMES))
    substitute = train_sub_no_augmn(data=data, target_model=target, sess=sess)
    #labels = [np.argmax(label, axis=None, out=None) for label in labels]
    #substitute = train_sub(data_aug=2, target_model=target, sess=sess, x_sub=data, y_sub=labels, lmbda=.1)

    print("Evaluating the accuracy of the substitute model on clean examples...")
    test_generator = TestGenerator(DATASET_PATH, BATCH_SIZE, IMAGE_SIZE, TEST_SAMPLES_NAMES)
    evaluate_generator(substitute.model, test_generator, EVAL_BATCH_SIZE)

    print("Evaluating the accuracy of the black-box model on clean examples...")
    evaluate_generator(target.model, test_generator, EVAL_BATCH_SIZE)

    print("Generating adversarial samples...")
    generate_adv_samples(substitute, test_generator, sess)

    print("Loading adversarial samples...")
    result_generator = TestGenerator(ADV_DATASET_PATH, BATCH_SIZE, IMAGE_SIZE, TEST_SAMPLES_NAMES)

    print("Evaluating the accuracy of the substitute model on adversarial examples...")
    evaluate_generator(substitute.model, result_generator, EVAL_BATCH_SIZE)

    print("Evaluating the accuracy of the black-box model on adversarial examples...")
    evaluate_generator(target.model, result_generator, EVAL_BATCH_SIZE)


def main():
    sess = tf.Session()
    keras.backend.set_session(sess)
    print("Session initialized")
    blackbox(sess)


if __name__ == '__main__':
    main()
