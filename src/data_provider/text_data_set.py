import copy
from . import DataSet


class TextDataSet(DataSet):
    """
        Implement a dataset class providing the image and it's corresponding text.
    """
    def __init__(self, images, labels, imagenames, shuffle=None, normalization=None):
        """
        Arguments:
            :param images: image datasets [nums, H, W, C] 4D ndarray
            :param labels: label dataset [nums, :] 2D ndarray
            :param shuffle: if need shuffle the dataset, 'once_prior_train' represent shuffle only once before training
                            'every_epoch' represent shuffle the data every epoch
            :param imagenames:
            :param normalization: if need do normalization to the dataset,
                                  'None': no any normalization
                                  'divide_255': divide all pixels by 255
                                  'divide_256': divide all pixels by 256
        """
        super(TextDataSet, self).__init__()

        self.__normalization = normalization
        if self.__normalization not in [None, 'divide_255', 'divide_256']:
            raise ValueError('normalization parameter wrong')
        self.__images = self.normalize_images(images, self.__normalization)

        self.__labels = labels
        self.__imagenames = imagenames
        self._epoch_images = copy.deepcopy(self.__images)
        self._epoch_labels = copy.deepcopy(self.__labels)
        self._epoch_imagenames = copy.deepcopy(self.__imagenames)

        self.__shuffle = shuffle
        if self.__shuffle not in [None, 'once_prior_train', 'every_epoch']:
            raise ValueError('shuffle parameter wrong')
        if self.__shuffle == 'every_epoch' or 'once_prior_train':
            self._epoch_images, self._epoch_labels, self._epoch_imagenames = self.shuffle_images_labels(
                self._epoch_images, self._epoch_labels, self._epoch_imagenames
            )
        self.__batch_counter = 0
        return

    @property
    def num_examples(self) -> int:
        assert self.__images.shape[0] == self.__labels.shape[0]
        return self.__labels.shape[0]

    @property
    def images(self):
        return self._epoch_images

    @property
    def labels(self):
        return self._epoch_labels

    @property
    def imagenames(self):
        return self._epoch_imagenames

    def next_batch(self, batch_size: int):
        start = self.__batch_counter * batch_size
        end = (self.__batch_counter + 1) * batch_size
        self.__batch_counter += 1
        images_slice = self._epoch_images[start:end]
        labels_slice = self._epoch_labels[start:end]
        imagenames_slice = self._epoch_imagenames[start:end]
        # if overflow restart from the beginning
        if images_slice.shape[0] != batch_size:
            self.__start_new_epoch()
            return self.next_batch(batch_size)
        else:
            return images_slice, labels_slice, imagenames_slice

    def __start_new_epoch(self):
        self.__batch_counter = 0
        if self.__shuffle == 'every_epoch':
            self._epoch_images, self._epoch_labels, self._epoch_imagenames = self.shuffle_images_labels(
                self._epoch_images, self._epoch_labels, self._epoch_imagenames)
