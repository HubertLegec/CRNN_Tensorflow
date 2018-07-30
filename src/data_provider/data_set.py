import numpy as np


class DataSet:
    """
        Implement some global useful functions used in all dataset
    """

    @staticmethod
    def shuffle_images_labels(images, labels, imagenames):
        images = np.array(images)
        labels = np.array(labels)

        assert images.shape[0] == labels.shape[0]

        random_index = np.random.permutation(images.shape[0])
        shuffled_images = images[random_index]
        shuffled_labels = labels[random_index]
        shuffled_imagenames = imagenames[random_index]

        return shuffled_images, shuffled_labels, shuffled_imagenames

    @staticmethod
    def normalize_images(images, normalization_type):
        """
        Arguments:
            images: numpy 4D array
            normalization_type: `str`, available choices:
                - divide_255
                - divide_256
                - by_chanels
        """
        if normalization_type == 'divide_255':
            images = images / 255
        elif normalization_type == 'divide_256':
            images = images / 256
        elif normalization_type is None:
            pass
        else:
            raise Exception("Unknown type of normalization")
        return images

    def normalize_all_images_by_chanels(self, initial_images):
        new_images = np.zeros(initial_images.shape)
        for i in range(initial_images.shape[0]):
            new_images[i] = self.normalize_image_by_chanel(initial_images[i])
        return new_images

    @staticmethod
    def normalize_image_by_chanel(image):
        new_image = np.zeros(image.shape)
        for chanel in range(3):
            mean = np.mean(image[:, :, chanel])
            std = np.std(image[:, :, chanel])
            new_image[:, :, chanel] = (image[:, :, chanel] - mean) / std
        return new_image

    def num_examples(self):
        raise NotImplementedError

    def next_batch(self, batch_size: int):
        raise NotImplementedError
