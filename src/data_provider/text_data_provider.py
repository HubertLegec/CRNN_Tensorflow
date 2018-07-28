import os.path as ops
import numpy as np
import cv2
from . import TextDataset


class TextDataProvider(object):
    """
        Implement the text data provider for training and testing the shadow net
    """
    def __init__(self, dataset_dir, annotation_name, validation_set=None, validation_split=None, shuffle=None,
                 normalization=None):
        """

        :param dataset_dir: str, where you save the dataset one class on folder
        :param annotation_name: annotation name
        :param validation_set:
        :param validation_split: `float` or None float: chunk of `train set` will be marked as `validation set`.
                                 None: if 'validation set' == True, `validation set` will be
                                 copy of `test set`
        :param shuffle: if need shuffle the dataset, 'once_prior_train' represent shuffle only once before training
                        'every_epoch' represent shuffle the data every epoch
        :param normalization: if need do normalization to the dataset,
                              'None': no any normalization
                              'divide_255': divide all pixels by 255
                              'divide_256': divide all pixels by 256
                              'by_chanels': substract mean of every chanel and divide each
                                            chanel data by it's standart deviation
        """
        self.__dataset_dir = dataset_dir
        self.__validation_split = validation_split
        self.__shuffle = shuffle
        self.__normalization = normalization
        self.__train_dataset_dir = ops.join(self.__dataset_dir, 'Train')
        self.__test_dataset_dir = ops.join(self.__dataset_dir, 'Test')

        assert ops.exists(dataset_dir)
        assert ops.exists(self.__train_dataset_dir)
        assert ops.exists(self.__test_dataset_dir)

        # add test dataset
        test_anno_path = ops.join(self.__test_dataset_dir, annotation_name)
        assert ops.exists(test_anno_path)

        with open(test_anno_path, 'r') as anno_file:
            info = np.array([tmp.strip().split() for tmp in anno_file.readlines()])

            test_images_org = [cv2.imread(ops.join(self.__test_dataset_dir, tmp), cv2.IMREAD_COLOR)
                               for tmp in info[:, 0]]
            test_images = np.array([cv2.resize(tmp, (100, 32)) for tmp in test_images_org])

            test_labels = np.array([tmp for tmp in info[:, 1]])

            test_imagenames = np.array([ops.basename(tmp) for tmp in info[:, 0]])

            self.test = TextDataset(test_images, test_labels, imagenames=test_imagenames,
                                    shuffle=shuffle, normalization=normalization)
        anno_file.close()

        # add train and validation dataset
        train_anno_path = ops.join(self.__train_dataset_dir, annotation_name)
        assert ops.exists(train_anno_path)

        with open(train_anno_path, 'r') as anno_file:
            info = np.array([tmp.strip().split() for tmp in anno_file.readlines()])

            train_images_org = [cv2.imread(ops.join(self.__train_dataset_dir, tmp), cv2.IMREAD_COLOR) for tmp in info[:, 0]]
            train_images = np.array([cv2.resize(tmp, (100, 32)) for tmp in train_images_org])

            train_labels = np.array([tmp for tmp in info[:, 1]])
            train_imagenames = np.array([ops.basename(tmp) for tmp in info[:, 0]])

            if validation_set is not None and validation_split is not None:
                split_idx = int(train_images.shape[0] * (1 - validation_split))
                self.train = TextDataset(images=train_images[:split_idx], labels=train_labels[:split_idx],
                                         shuffle=shuffle, normalization=normalization,
                                         imagenames=train_imagenames[:split_idx])
                self.validation = TextDataset(images=train_images[split_idx:], labels=train_labels[split_idx:],
                                              shuffle=shuffle, normalization=normalization,
                                              imagenames=train_imagenames[split_idx:])
            else:
                self.train = TextDataset(images=train_images, labels=train_labels, shuffle=shuffle,
                                         normalization=normalization, imagenames=train_imagenames)

            if validation_set and not validation_split:
                self.validation = self.test
        anno_file.close()
        return

    def __str__(self):
        provider_info = 'Dataset_dir: {:s} contain training images: {:d} validation images: {:d} testing images: {:d}'.\
            format(self.__dataset_dir, self.train.num_examples, self.validation.num_examples, self.test.num_examples)
        return provider_info

    @property
    def dataset_dir(self):
        return self.__dataset_dir

    @property
    def train_dataset_dir(self):
        return self.__train_dataset_dir

    @property
    def test_dataset_dir(self):
        return self.__test_dataset_dir
