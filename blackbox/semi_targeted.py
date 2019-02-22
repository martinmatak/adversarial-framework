import tensorflow as tf
import keras
import numpy as np
from keras.optimizers import Adam
from pathlib import Path

from six.moves import xrange

from keras.models import load_model

from cleverhans.attacks import FastGradientMethod,  CarliniWagnerL2
from cleverhans.attacks_tf import jacobian_graph, jacobian_augmentation
from cleverhans.utils_keras import KerasModelWrapper


from whitebox.attacks import fgsm, cw
from utils.image_ops import L2_distance, save_image, resize_images
from utils.numpy_ops import convert_to_one_hot
from utils.generator import TestGenerator, TransferGenerator, CustomGenerator
from utils.model_ops import evaluate_generator, age_mae, get_dataset, model_argmax, get_simple_model, get_model, save_model

# prototype constants
MODEL_PATH = '/Users/mmatak/dev/thesis/adversarial_framework/model/resnet50-3.436-5.151-sgd.hdf5'
TRAINING_SET_PATH = '/Users/mmatak/dev/thesis/datasets/appa-real-release-100'
TEST_SET_PATH = '/Users/mmatak/dev/thesis/datasets/appa-real-release-1'
NUM_EPOCHS_SUB = 1
ADV_ID_START = 5615
ADV_ID_END = 7613
NB_SUB_CLASSES = 5
AUG_BATCH_SIZE = 512
PREDICT_BATCH_SIZE = 64

# remote constants
#MODEL_PATH = '/root/age-estimation/checkpoints/resnet50-3.436-5.151-sgd.hdf5'
#TRAINING_SET_PATH = '/root/datasets/appa-real-release'
#TEST_SET_PATH = '/root/datasets/appa-real-release-100'
#ADV_ID_START = 5613

ATTACK_NAME = 'fgsm'
RESULT_PATH = TEST_SET_PATH + '-adv/blackbox/' + ATTACK_NAME + '/'
CSV_PATH = TRAINING_SET_PATH + '/' + 'custom-dataset.csv'

BATCH_SIZE = 1
EVAL_BATCH_SIZE = 1
IMAGE_SIZE_BBOX = 224
IMAGE_SIZE_SUB = 32
NUM_OF_CHANNELS = 3
NB_CLASSES = 101


def prep_bbox():
    model = load_model(MODEL_PATH, compile=False)
    model.compile(optimizer=Adam(), loss="categorical_crossentropy", metrics=[age_mae])
    wrap = KerasModelWrapper(model)
    print("Model loaded")
    return wrap


def post_process_predictions(predictions):
    post_process_predictions = np.ones_like(predictions)
    for index, label in enumerate(predictions):
        #print("label: ", label)
        new_label = int(min(label, 100) / int(101/NB_SUB_CLASSES))
        #print("new label: ", new_label)
        post_process_predictions[index] = new_label
    return post_process_predictions


def bbox_predict(model, data, sess, x, batch_size=PREDICT_BATCH_SIZE):
    # here comes API call or anything similar
    print("querying bbox...")
    data = resize_images(data, IMAGE_SIZE_BBOX)
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
              x_sub, y_sub, lmbda, target_model, aug_batch_size=AUG_BATCH_SIZE):
    placeholder_sub = tf.placeholder(tf.float32, shape=(None, IMAGE_SIZE_SUB, IMAGE_SIZE_SUB, NUM_OF_CHANNELS))
    placeholder_bbox = tf.placeholder(tf.float32, shape=(None, IMAGE_SIZE_BBOX, IMAGE_SIZE_BBOX, NUM_OF_CHANNELS))

    print("Loading substitute model...")
    model = get_simple_model(NB_SUB_CLASSES)
    model.compile(optimizer=Adam(), loss="categorical_crossentropy", metrics=['accuracy'])
    model_sub = KerasModelWrapper(model)

    preds_sub = model_sub.get_logits(placeholder_sub)
    print("Subsitute model loaded.")

    # Define the Jacobian symbolically using TensorFlow
    print("Defining jacobian graph...")
    grads = jacobian_graph(preds_sub, placeholder_sub, NB_SUB_CLASSES)
    print("Jacobian graph defined.")

    train_gen = TransferGenerator(x_sub, y_sub, num_classes=NB_SUB_CLASSES, batch_size=BATCH_SIZE, image_size=IMAGE_SIZE_SUB, encoding_needed=False)
    for rho in xrange(data_aug):
        print("Substitute training epoch #" + str(rho))
        train_gen.reinitialize(x_sub, y_sub, BATCH_SIZE, IMAGE_SIZE_SUB, encoding_needed=False)
        model_sub.model.fit_generator(generator=train_gen, epochs=NUM_EPOCHS_SUB)

        print("Saving substitute model that is trained so far")
        path = Path(__file__).resolve().parent.parent.joinpath("model")
        save_model(str(path) + "/sub_model_after_epoch" + str(rho) + ".h5", model_sub)

        # input_sample = np.empty(shape=(1, IMAGE_SIZE_SUB, IMAGE_SIZE_SUB, NUM_OF_CHANNELS), dtype=np.float32)
        if rho < data_aug - 1:
            print("Augmenting substitute training data...")
            # Perform the Jacobian augmentation
            lmbda_coef = 2 * int(int(rho / 3) != 0) - 1

            # per image augmentation
            # x_sub_tmp = np.vstack([x_sub, x_sub])
            # for i in range(0, len(y_sub)):
            #     input_sample[0, :, :, :] = x_sub[i]
            #     adv = jacobian_augmentation(
            #         sess=sess,
            #         x=placeholder_sub,
            #         X_sub_prev=input_sample,
            #         Y_sub=[y_sub[i]],
            #         grads=grads,
            #         lmbda=lmbda_coef*lmbda,
            #         aug_batch_size=aug_batch_size
            #     )
            #     x_sub_tmp[2*i] = adv[0, :, :, :]
            #     x_sub_tmp[2*i + 1] = adv[1, :, :, :]
            #
            # x_sub = x_sub_tmp

            x_sub = jacobian_augmentation(sess, placeholder_sub, x_sub, y_sub, grads,
                                          lmbda_coef * lmbda, aug_batch_size)
            print("Substitute training data augmented.")

            print("Labeling substitute training data using bbox...")
            y_sub = np.hstack([y_sub, y_sub])
            x_sub_prev = x_sub[int(len(x_sub) / 2):]
            y_sub[int(len(x_sub)/2):] = bbox_predict(target_model, x_sub_prev, sess, placeholder_bbox)
    return model_sub


def train_sub_no_augmn(data, target_model, sess):
    print("Loading a substitute model...")

    x = tf.placeholder(tf.float32, shape=(None, IMAGE_SIZE_SUB, IMAGE_SIZE_SUB, NUM_OF_CHANNELS))

    model = get_model()
    model.compile(optimizer=Adam(), loss="categorical_crossentropy", metrics=[age_mae])
    model_sub = KerasModelWrapper(model)

    print("Substitute model loaded")

    print("Labeling samples...")
    labels = bbox_predict(target_model, data, sess, x, batch_size=1)
    print("Samples labeled")

    print("Training a substitute model...")
    train_gen = TransferGenerator(data, labels , BATCH_SIZE, IMAGE_SIZE_SUB, encoding_needed=False)
    model_sub.model.fit_generator(generator=train_gen, epochs=NUM_EPOCHS)
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

    img_ids = [str("00" + str(i)) for i in range(ADV_ID_START, ADV_ID_END)]
    id_index = 0

    TEN_LABEL = convert_to_one_hot(0, NB_SUB_CLASSES)
    NINETY_LABEL = convert_to_one_hot(NB_SUB_CLASSES-1, NB_SUB_CLASSES)
    for legit_sample, legit_label in generator:

        ground_truth = np.argmax(legit_label)

        if ground_truth > 50:
            adv_x = attack_instance.attack(legit_sample, None, attack_instance_graph)
        else:
            adv_x = attack_instance.attack(legit_sample, None, attack_instance_graph)

        diff_L2.append(L2_distance(legit_sample, adv_x))

        save_image(RESULT_PATH + '/test/' + img_ids[id_index] + ".jpg_face.jpg", adv_x[0, :, :, :])
        id_index += 1

    print("Average L2 perturbation summed by channels: ", str(sum(diff_L2) / float(len(diff_L2))))


def assert_equal(L1, L2):
    return len(L1) == len(L2) and sorted(L1) == sorted(L2)


def blackbox(sess):
    # simulate the black-box model locally
    print("Preparing the black-box model.")
    target = prep_bbox()

    # train substitute using method from https://arxiv.org/abs/1602.02697
    print("Training the substitute model by querying the target network..")
    data, labels = get_dataset(CustomGenerator(CSV_PATH, NB_SUB_CLASSES, BATCH_SIZE, IMAGE_SIZE_SUB))
    labels = [np.argmax(label, axis=None, out=None) for label in labels]
    substitute = train_sub(data_aug=6, target_model=target, sess=sess, x_sub=data, y_sub=labels, lmbda=.1)

    print("Evaluating the accuracy of the black-box model on clean examples...")
    bbox_generator = TestGenerator(TEST_SET_PATH, BATCH_SIZE, IMAGE_SIZE_BBOX)
    evaluate_generator(target.model, bbox_generator, EVAL_BATCH_SIZE)

    print("Evaluating the accuracy of the substitute model on clean examples...")
    test_data, test_labels = get_dataset(bbox_generator)
    test_labels = [np.argmax(label, axis=None, out=None) for label in test_labels]
    test_labels = [int(label / int(101/NB_SUB_CLASSES)) for label in test_labels]
    sub_generator = TransferGenerator(test_data, test_labels, NB_SUB_CLASSES, BATCH_SIZE, IMAGE_SIZE_SUB)
    evaluate_generator(substitute.model, sub_generator, EVAL_BATCH_SIZE)

    print("Generating adversarial samples...")
    generate_adv_samples(substitute, sub_generator, sess)

    print("Loading adversarial samples...")
    result_bbox_generator = TestGenerator(RESULT_PATH, BATCH_SIZE, IMAGE_SIZE_BBOX)

    print("Evaluating the accuracy of the substitute model on adversarial examples...")
    result_data, result_labels = get_dataset(result_bbox_generator)

    result_labels = [np.argmax(label, axis=None, out=None) for label in result_labels]
    result_labels = [int(label / int(101/NB_SUB_CLASSES)) for label in result_labels]

    assert_equal(result_labels, test_labels)
    sub_adv_generator = TransferGenerator(result_data, result_labels, NB_SUB_CLASSES, BATCH_SIZE, IMAGE_SIZE_SUB)
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
