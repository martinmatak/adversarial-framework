import numpy as np
from keras.applications import ResNet50, InceptionResNetV2, VGG16, VGG19
from keras.layers import Dense
from keras.models import Model
from keras import backend as K
from keras.optimizers import Adam
from keras.utils import to_categorical


def model_argmax(sess, x, predictions, samples, feed=None):
    """
    Helper function that computes the current class prediction
    :param sess: TF session
    :param x: the input placeholder
    :param predictions: the model's symbolic output
    :param samples: numpy array with input samples (dims must match x)
    :param feed: An optional dictionary that is appended to the feeding
             dictionary before the session runs. Can be used to feed
             the learning phase of a Keras model for instance.
    :return: the argmax output of predictions, i.e. the current predicted class
    """
    feed_dict = {x: samples}
    if feed is not None:
        feed_dict.update(feed)
    probabilities = sess.run(predictions, feed_dict)

    if samples.shape[0] == 1:
        return np.argmax(probabilities)
    else:
        return np.argmax(probabilities, axis=1)


def get_dataset(generator):
    # load dataset
    x_tmp, y_tmp = zip(*(generator[i] for i in range(len(generator))))
    x_test, y_test = np.vstack(x_tmp), np.vstack(y_tmp)
    print("Dataset loaded")

    return x_test, y_test

def evaluate_generator(model, generator, batch_size):
    x_test, y_test = get_dataset(generator)
    evaluate(model, x_test, y_test, batch_size)


def evaluate(model, x_test, y_test, batch_size):
    result = model.evaluate(x_test, y_test, batch_size, verbose=1)

    print(model.metrics_names[1] + ": " + str(result[1]))


def age_mae(y_true, y_pred):
    true_age = K.sum(y_true * K.arange(0, 101, dtype="float32"), axis=-1)
    pred_age = K.sum(y_pred * K.arange(0, 101, dtype="float32"), axis=-1)
    mae = K.mean(K.abs(true_age - pred_age))
    return mae


def age_mae(y_true, y_pred):
    true_age = K.sum(y_true * K.arange(0, 101, dtype="float32"), axis=-1)
    pred_age = K.sum(y_pred * K.arange(0, 101, dtype="float32"), axis=-1)
    mae = K.mean(K.abs(true_age - pred_age))
    return mae


def get_model(model_name="ResNet50"):
    base_model = None

    if model_name == "ResNet50":
        base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3), pooling="avg")
    elif model_name == "InceptionResNetV2":
        base_model = InceptionResNetV2(include_top=False, weights='imagenet', input_shape=(299, 299, 3), pooling="avg")
    elif model_name == "VGG16":
        base_model = VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3), pooling="avg")
    elif model_name == "VGG19":
        base_model = VGG19(include_top=False, weights='imagenet', input_shape=(224, 224, 3), pooling="avg")

    prediction = Dense(units=101, kernel_initializer="he_normal", use_bias=False, activation="softmax",
                       name="pred_age")(base_model.output)

    model = Model(inputs=base_model.input, outputs=prediction)

    return model


def get_compiled_model(model_name="ResNet50", optimizer=Adam()):
    model = get_model(model_name)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=[age_mae])
    return model


def train_model(model, data, labels, nb_classes):
    #TODO
    return model
    # data = to_categorical(data, nb_classes)
    # model.fit(x=data, y=labels, batch_size=1)
    # return model