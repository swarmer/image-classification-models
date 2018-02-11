import pickle
import random

import numpy as np


HEIGHT = 32
WIDTH = 32
IMAGE_SIZE = HEIGHT * WIDTH * 3
LABEL_COUNT = 10


class Dataset:
    def __init__(self, pairs):
        # (image tensor, label)
        self.pairs = pairs


class Experiment:
    def __init__(self, train_set, test_set):
        self.train_set = train_set
        self.test_set = test_set


def _onehot_encode(x, label_count, normalize=True):
    arr = [0 if not normalize else 0.01] * label_count
    arr[x] = 1 if not normalize else 0.99
    return arr


def _prepare_image(img, normalize=True):
    img = img.reshape((3, HEIGHT, WIDTH)).transpose((1, 2, 0))

    if normalize:
        img = img / 255.0 * 0.99 + 0.01

    return img


def read_pairs(paths, shuffle=True, normalize=True):
    for path in paths:
        with open(path, 'rb') as batch_file:
            batch = pickle.load(batch_file, encoding='latin1')

            images = (
                np.array([_prepare_image(img) for img in batch['data']])
                .reshape((-1, IMAGE_SIZE))
            )
            labels = (
                np.array([_onehot_encode(l, LABEL_COUNT, normalize) for l in batch['labels']])
                .reshape((-1, LABEL_COUNT))
            )

            if shuffle:
                indexes = np.random.permutation(len(images))
                images = images[indexes]
                labels = labels[indexes]

            yield images, labels


def read_dataset(paths, normalize=True):
    return Dataset(read_pairs(paths, normalize=normalize))
