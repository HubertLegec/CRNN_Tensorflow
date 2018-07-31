from unittest import TestCase
from numpy.testing import assert_allclose
from . import calculate_mean_accuracy, get_batch_accuracy


class CrnnTesterTest(TestCase):
    def test_batch_accuracy(self):
        labels = ["dog", "home", "surgery"]
        predictions = ["dog", "home", "surerry"]
        accuracy = get_batch_accuracy(predictions, labels)
        assert_allclose(accuracy, [1.0, 1.0, 0.714], atol=0.001)

    def test_mean_accuracy(self):
        accuracies = [0.2, 0.4, 0.3]
        mean_accuracy = calculate_mean_accuracy(accuracies)
        assert_allclose(mean_accuracy, 0.3, atol=0.01)
