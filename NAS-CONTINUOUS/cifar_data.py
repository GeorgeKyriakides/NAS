'''
    Script to load the cifar dataset into a Dataset class
    Adapted from https://github.com/exelban/tensorflow-cifar-10/blob/master/include/data.py
'''

import pickle
import numpy as np
import os
from urllib.request import urlretrieve
import tarfile
import zipfile
import sys


class Dataset(object):
    def __init__(self, x, y, labels, shuffle=True):
        self.num_examples = len(x)
        self.total_given = 0
        self.shuffle = shuffle
        self.indices = np.array([x for x in range(self.num_examples)])
        self.x = x
        self.y = y

    def next_batch(self, batch_size):
        if self.total_given == 0 and self.shuffle:
            np.random.shuffle(self.indices)
        start = self.total_given
        end = min(start+batch_size, self.num_examples-1)
        self.total_given = end if end < self.num_examples-1 else 0
        inds = self.indices[start:end]

        x = self.x[inds]
        y = self.y[inds]

        return x, y


class Datasets(object):
    def __init__(self, train, test):
        self.train = train
        self.test = test


def load_cifar10(reshape=True, shuffle=True):
    train_x, train_y, train_l = get_data_set(
        name="train", cifar=10, reshape=reshape)

    test_x, test_y, test_l = get_data_set(
        name="test", cifar=10, reshape=reshape)

    train = Dataset(train_x, train_y, train_l, shuffle)
    test = Dataset(test_x, test_y, test_l, shuffle)
    return Datasets(train, test)


def get_data_set(name="train", cifar=10, reshape=True):
    x = None
    y = None
    labels = None

    maybe_download_and_extract()

    folder_name = "cifar_10" if cifar == 10 else "cifar_100"

    f = open('./data_set/'+folder_name+'/batches.meta', 'rb')
    datadict = pickle.load(f, encoding='latin1')
    f.close()
    labels = datadict['label_names']

    if name is "train":
        for i in range(5):
            f = open('./data_set/'+folder_name +
                     '/data_batch_' + str(i + 1), 'rb')
            datadict = pickle.load(f, encoding='latin1')
            f.close()

            _X = datadict["data"]
            _Y = datadict['labels']

            _X = np.array(_X, dtype=float) / 255.0
            _X = _X.reshape([-1, 3, 32, 32])
            _X = _X.transpose([0, 2, 3, 1])
            if reshape:
                _X = _X.reshape(-1, 32*32*3)

            if x is None:
                x = _X
                y = _Y
            else:
                x = np.concatenate((x, _X), axis=0)
                y = np.concatenate((y, _Y), axis=0)

    elif name is "test":
        f = open('./data_set/'+folder_name+'/test_batch', 'rb')
        datadict = pickle.load(f, encoding='latin1')
        f.close()

        x = datadict["data"]
        y = np.array(datadict['labels'])

        x = np.array(x, dtype=float) / 255.0
        x = x.reshape([-1, 3, 32, 32])
        x = x.transpose([0, 2, 3, 1])
        if reshape:
            x = x.reshape(-1, 32*32*3)

    def dense_to_one_hot(labels_dense, num_classes=10):
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1

        return labels_one_hot

    return x, dense_to_one_hot(y), labels



def _print_download_progress(count, block_size, total_size):
    pct_complete = float(count * block_size) / total_size
    msg = "\r- Download progress: {0:.1%}".format(pct_complete)
    sys.stdout.write(msg)
    sys.stdout.flush()


def maybe_download_and_extract():
    main_directory = "./data_set/"
    cifar_10_directory = main_directory+"cifar_10/"
    if not os.path.exists(main_directory):
        os.makedirs(main_directory)

        url = "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
        filename = url.split('/')[-1]
        file_path = os.path.join(main_directory, filename)
        zip_cifar_10 = file_path
        file_path, _ = urlretrieve(url=url, filename=file_path,
                                   reporthook=_print_download_progress)

        print()
        print("Download finished. Extracting files.")
        if file_path.endswith(".zip"):
            zipfile.ZipFile(file=file_path, mode="r").extractall(
                main_directory)
        elif file_path.endswith((".tar.gz", ".tgz")):
            tarfile.open(name=file_path, mode="r:gz").extractall(
                main_directory)
        print("Done.")

        os.rename(main_directory+"./cifar-10-batches-py", cifar_10_directory)
        os.remove(zip_cifar_10)
