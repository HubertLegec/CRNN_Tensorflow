from unittest import TestCase
from numpy.testing import assert_allclose
from . import calculate_array_mean, get_batch_accuracy


class CrnnTesterTest(TestCase):
    def test_batch_accuracy(self):
        labels = ["dog", "home", "surgery"]
        predictions = ["dog", "home", "surerry"]
        accuracy = get_batch_accuracy(predictions, labels)
        assert_allclose(accuracy, [1.0, 1.0, 0.714], atol=0.001)

    def test_not_equal_lengths(self):
        labels = ["home", "loss"]
        predictions = ["homea", "los"]
        accuracy = get_batch_accuracy(predictions, labels)
        assert_allclose(accuracy, [1, 0.75], atol=0.001)

    def test_empty_gt_empty_prediction(self):
        labels = [""]
        predictions = [""]
        accuracy = get_batch_accuracy(predictions, labels)
        assert_allclose(accuracy, [1], atol=0.001)

    def test_empty_gt_not_empty_prediction(self):
        labels = [""]
        predictions = ["pre"]
        accuracy = get_batch_accuracy(predictions, labels)
        assert_allclose(accuracy, [0], atol=0.001)

    def test_mean_accuracy(self):
        accuracies = [0.2, 0.4, 0.3]
        mean_accuracy = calculate_array_mean(accuracies)
        assert_allclose(mean_accuracy, 0.3, atol=0.01)
