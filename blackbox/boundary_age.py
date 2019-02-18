import foolbox
import keras
import numpy as np
from utils.model_ops import age_mae
from utils.image_ops import save_image, load_image
from foolbox.models.base import Model
from keras.models import load_model
from keras.optimizers import Adam
import matplotlib
import matplotlib.pyplot as plt


BATCH_SIZE = 1
MODEL_PATH = '/Users/mmatak/dev/thesis/adversarial_framework/model/resnet50-3.436-5.151-sgd.hdf5'
IMAGE_SOURCE_PATH = '/Users/mmatak/dev/thesis/datasets/chosen-images/10yrs-girl.jpg'
IMAGE_TARGET_PATH = '/Users/mmatak/dev/thesis/datasets/chosen-images/donald-trump.jpg'
IMAGE_SIZE = 224
NUM_ITERATIONS = 10000
NB_CLASSES = 101

class WebsiteModel(Model):

    def __init__(
            self,
            model,
            bounds,
            num_classes,
            channel_axis=1,
            preprocessing=(0, 1)):

        super(WebsiteModel, self).__init__(bounds=bounds,
                                           channel_axis=channel_axis,
                                           preprocessing=preprocessing)
        self._model = model
        self._num_classes = num_classes
        self._num_queried = 0

    def compute_max_probability(self, logits_values, image_name):
        exps = [np.exp(i) for i in logits_values]
        sum_of_exps = sum(exps)
        softmax = [j / sum_of_exps for j in exps]
        x = []
        y = []

        for age in range(0, 101):
            x.append(age)
            y.append(softmax[age])

        fig = plt.figure()
        fig.suptitle('Confidence graph', fontsize=20)
        plt.xlabel('age', fontsize=14)
        plt.ylabel('probability', fontsize=14)

        plt.plot([x], [y], 'b.')
        plt.axis([0, 100, 0, max(y) * 1.2])

        plt.savefig("/Users/mmatak/dev/thesis/adversarial_framework/results/softmax" + image_name + ".png",
                    bbox_inches='tight')
        plt.close(fig)

        return softmax[int(np.argmax(softmax))]

    def batch_predictions(self, images):
        preds = self._model.batch_predictions(images)
        self._num_queried += 1
        if self._num_queried % 1000 == 0:
            save_image("/Users/mmatak/dev/thesis/adversarial_framework/results/image" + str(self._num_queried) + ".jpg",
                       images[0][:, :, :])
            self.compute_max_probability(preds[0], str(self._num_queried))
            print("num of queries: " + str(self._num_queried))
        # GET PREDICTIONS OF WEBSITE HERE (should be like logits [shape = batch size, num_classes] but can be zero-one)
        return preds

    def num_classes(self):
        return self._num_classes

def prep_bbox():
    model = load_model(MODEL_PATH, compile=False)
    model.compile(optimizer=Adam(), loss="categorical_crossentropy", metrics=[age_mae])
    return model


# instantiate model
keras.backend.set_learning_phase(0)
fmodel = foolbox.models.KerasModel(prep_bbox(), bounds=(0, 255))
fmodel = WebsiteModel(fmodel, fmodel._bounds, NB_CLASSES)

image_source = load_image(IMAGE_SOURCE_PATH, IMAGE_SIZE)
image_target= load_image(IMAGE_TARGET_PATH, IMAGE_SIZE)


image_source = image_source.astype(np.float32)
image_target = image_target.astype(np.float32)

# obtain target label
preds = fmodel.predictions(image_target)
label_target = int(np.argmax(preds))

print("target label: " + str(label_target))
print("confidence: " + str(fmodel.compute_max_probability(preds, "benign")))

attack = foolbox.attacks.BoundaryAttack(fmodel, criterion=foolbox.criteria.TargetClass(label_target))
adversarial = attack(input_or_adv=image_source, label=1, starting_point=image_target, verbose=True, iterations=NUM_ITERATIONS)

save_image("/Users/mmatak/dev/thesis/adversarial_framework/results/final_adversarial" + str(NUM_ITERATIONS) + ".jpg", adversarial[:, :, :])

# if the attack fails, adversarial will be None and a warning will be printed
preds_adv = fmodel.predictions(adversarial)
print("label of adversarial sample: " + str(int(np.argmax(preds_adv))))
print("confidence: " + str(fmodel.compute_max_probability(preds_adv, "adversarial")))

import matplotlib.pyplot as plt

plt.figure()

plt.subplot(1, 4, 1)
plt.title('Original')
plt.imshow(image_source[:, :, ::-1] / 255)  # division by 255 to convert [0, 255] to [0, 1]
plt.axis('off')

plt.subplot(1, 4, 2)
plt.title('Adversarial')
plt.imshow(adversarial[:, :, ::-1] / 255)  # ::-1 to convert BGR to RGB
plt.axis('off')

plt.subplot(1, 4, 3)
plt.title('Target')
plt.imshow(image_target[:, :, ::-1] / 255)  # division by 255 to convert [0, 255] to [0, 1]
plt.axis('off')

plt.subplot(1, 4, 4)
plt.title('Difference')
difference = adversarial[:, :, ::-1] - image_source
plt.imshow(difference / abs(difference).max() * 0.2 + 0.5)
plt.axis('off')
plt.show()