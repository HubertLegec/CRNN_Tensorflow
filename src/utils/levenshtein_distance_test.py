from unittest import TestCase
from . import levenshtein_distance


class LevenhsteinDistanceTest(TestCase):
    def test_empty_strings(self):
        s1 = ""
        s2 = ""
        distance = levenshtein_distance(s1, s2)
        assert distance == 0

    def test_equal_strings(self):
        s1 = "dog"
        s2 = "dog"
        distance = levenshtein_distance(s1, s2)
        assert distance == 0

    def test__different_strings_of_same_size(self):
        s1 = "surgery"
        s2 = "surerry"
        distance = levenshtein_distance(s1, s2)
        assert distance == 2

    def test_different_strings(self):
        s1 = "home"
        s2 = "humea"
        distance = levenshtein_distance(s1, s2)
        assert distance == 2
