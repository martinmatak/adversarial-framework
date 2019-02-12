import foolbox
import keras
import numpy as np
from utils.generator import TestGenerator
from utils.model_ops import age_mae
from utils.image_ops import L2_distance, save_image
from keras.models import load_model
from keras.optimizers import Adam

BATCH_SIZE = 1
TEST_SET_PATH = '/Users/mmatak/dev/thesis/datasets/appa-real-release-100'
MODEL_PATH = '/Users/mmatak/dev/thesis/adversarial_framework/model/resnet50-3.436-5.151-sgd.hdf5'
RESULT_PATH = TEST_SET_PATH + '-adv/blackbox/fgsm/'
IMAGE_SIZE = 224

def prep_bbox():
    model = load_model(MODEL_PATH, compile=False)
    model.compile(optimizer=Adam(), loss="categorical_crossentropy", metrics=[age_mae])
    return model

# instantiate model
keras.backend.set_learning_phase(0)
fmodel = foolbox.models.KerasModel(prep_bbox(), bounds=(0, 255))

test_generator = TestGenerator(TEST_SET_PATH, BATCH_SIZE, IMAGE_SIZE)


image_source, _ = test_generator[0]
image_target, _ = test_generator[2]


image_source = image_source[0].astype(np.float32)
image_target = image_target[0].astype(np.float32)

# obtain target label
label_source = int(np.argmax(fmodel.predictions(image_source)))
label_target = int(np.argmax(fmodel.predictions(image_target)))

print("source label: " + str(label_source))
print("target label: " + str(label_target))


attack = foolbox.attacks.BoundaryAttack(fmodel, criterion=foolbox.criteria.TargetClass(label_target))
adversarial = attack(input_or_adv=image_source, label=1, starting_point=image_target, verbose=True, iterations=5000)

save_image("/Users/mmatak/dev/thesis/adversarial_framework/results/image.jpg", adversarial[:, :, :])

# if the attack fails, adversarial will be None and a warning will be printed
print("label of adversarial sample: " + str(int(np.argmax(fmodel.predictions(adversarial)))))
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