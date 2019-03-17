import numpy as np
from keras import backend as K
from keras.utils import to_categorical
from keras.applications import ResNet50, VGG16, InceptionResNetV2
from keras.layers import Dense, Flatten, Activation, Conv2D
from keras.models import Model, Sequential

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
    return K.mean(K.abs(K.argmax(y_true, axis=-1) - K.argmax(y_pred, axis=-1)))


def train_model(model, data, labels, nb_classes):
    # return model
    labels = to_categorical(labels, nb_classes)
    model.fit(x=data, y=labels, batch_size=1, epochs=40)
    return model


def get_model(model_name, num_of_classes=101):
    base_model = None

    if model_name == "ResNet50":
        base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3), pooling="avg")
        output = base_model.output
    elif model_name == "VGG16":
        base_model = VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3), pooling="avg")
        x = base_model.output
        output = Dense(1024, activation='relu')(x)
        output = Dense(1024, activation='relu')(output)
    elif model_name == "InceptionResNetV2":
        base_model = InceptionResNetV2(include_top=False, weights='imagenet', input_shape=(299, 299, 3), pooling="avg")
        output = base_model.output
    else:
        print("model not supported")
        exit(1)

    prediction = Dense(units=num_of_classes, kernel_initializer="he_normal", use_bias=False, activation="softmax",
                       name="pred_age")(output)

    model = Model(inputs=base_model.input, outputs=prediction)
    return model

def get_simple_model(num_classes, image_size):
    nb_filters = 64
    model = Sequential()
    model.add(Conv2D(nb_filters, (8, 8), padding='same', input_shape=(image_size, image_size, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(nb_filters * 2, (6, 6), padding='valid'))
    model.add(Activation('relu'))
    model.add(Conv2D(nb_filters * 2, (5, 5),  padding='valid'))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    return model


def save_model(path, keras_model):
    '''
    saves the model with weights + architecture description
    :param path: path where to save the model
    :param model: instance of keras model
    '''
    print("saving model to path ", path)
    keras_model.save(path)
