from pathlib import Path
import numpy as np
import pandas as pd
import cv2
from keras.utils import Sequence, to_categorical


class TestGenerator(Sequence):

    def __init__(self, appa_dir, batch_size=32, image_size=224):
        self.image_path_and_age = []
        self._load_appa(appa_dir)
        self.image_num = len(self.image_path_and_age)
        self.batch_size = batch_size
        self.image_size = image_size

    def __len__(self):
        return self.image_num // self.batch_size

    def __getitem__(self, idx):
        batch_size = self.batch_size
        image_size = self.image_size
        x = np.zeros((batch_size, image_size, image_size, 3), dtype=np.uint8)
        y = np.zeros((batch_size, 1), dtype=np.int32)

        for i in range(batch_size):
            image_path, age = self.image_path_and_age[idx * batch_size + i]
            image = cv2.imread(str(image_path))
            x[i] = cv2.resize(image, (image_size, image_size))
            y[i] = age

        return x, to_categorical(y, 101)

    def _load_appa(self, appa_dir):
        appa_root = Path(appa_dir)
        val_image_dir = appa_root.joinpath("test")
        gt_val_path = appa_root.joinpath("gt_avg_test.csv")
        df = pd.read_csv(str(gt_val_path))

        for i, row in df.iterrows():
            age = min(99, int(row.apparent_age_avg))
            # age = int(row.real_age)
            image_path = val_image_dir.joinpath(row.file_name + "_face.jpg")

            if image_path.is_file():
                self.image_path_and_age.append([str(image_path), age])


class TransferGenerator(Sequence):

    def __init__(self, data, labels, num_classes=101, batch_size=32, image_size=224):
        # encoding needed is TRUE if given labels are one hot encoded
        self.data = data
        self.labels = labels
        self.image_num = len(labels)
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_classes = num_classes
        self.indices = np.arange(len(self.labels))
        self.on_epoch_end()

    def __len__(self):
        return self.image_num // self.batch_size

    def __getitem__(self, idx):
        batch_size = self.batch_size
        image_size = self.image_size
        x = np.zeros((batch_size, image_size, image_size, 3), dtype=np.uint8)
        y = np.zeros((batch_size, 1), dtype=np.int32)
        for i in range(batch_size):
            image = self.data[self.indices[idx*batch_size + i]]
            x[i] = cv2.resize(image, (image_size, image_size))
            label = self.labels[self.indices[idx*batch_size + i]]
            y[i] = label

        return x, to_categorical(y, num_classes=self.num_classes)


    def on_epoch_end(self):
        'Updates indexes after each epoch'
        np.random.shuffle(self.indices)


    def reinitialize(self, data, labels, batch_size=32, image_size=224):
        # encoding needed is TRUE if given labels are one hot encoded
        self.data = data
        self.labels = labels
        self.image_num = len(labels)
        self.batch_size = batch_size
        self.image_size = image_size


class CustomGenerator(Sequence):
    def __init__(self, csv_path, num_classes, batch_size=32, image_size=224):
        self.image_path_and_age = []
        self._load_csv(csv_path)
        self.image_num = len(self.image_path_and_age)
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_classes = num_classes

    def __len__(self):
        return self.image_num // self.batch_size

    def __getitem__(self, idx):
        batch_size = self.batch_size
        image_size = self.image_size
        x = np.zeros((batch_size, image_size, image_size, 3), dtype=np.uint8)
        y = np.zeros((batch_size, 1), dtype=np.int32)

        for i in range(batch_size):
            image_path, age = self.image_path_and_age[idx * batch_size + i]
            image = cv2.imread(str(image_path))
            x[i] = cv2.resize(image, (image_size, image_size))
            y[i] = self._convert_age(age)

        return x, to_categorical(y, self.num_classes)

    def _load_csv(self, csv_path):
        root_path = Path(csv_path).parent
        images_dir = root_path.joinpath("test")
        df = pd.read_csv(str(csv_path))
        for i, row in df.iterrows():
            age = min(99, int(row.apparent_age_avg))
            image_path = images_dir.joinpath(row.file_name + "_face.jpg")
            if image_path.is_file():
                self.image_path_and_age.append([str(image_path), age])

    def _convert_age(self, age):
        return int(min(age, 99) / int(101 / self.num_classes))
