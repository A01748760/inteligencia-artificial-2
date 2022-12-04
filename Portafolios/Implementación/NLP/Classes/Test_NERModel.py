"""
Author: David Rodriguez Fragoso
Class that tests the NER_Model class
Creation date: 29/11/2022
Last updated: 29/11/2022
"""

from NER_Model import NERModel
import unittest
import os


class Test(unittest.TestCase):

    def test_NER(self):
        print(f'{"TESTING NER":-^40}')
        n_examples = 600
        samples = len(NERModel.downsample(n_examples))
        self.assertEqual(n_examples, samples, "NOT PASSED")

        n_examples = 100
        samples = len(NERModel.downsample(n_examples))
        self.assertEqual(n_examples, samples, "NOT PASSED")

        n_examples = 1000
        samples = len(NERModel.downsample(n_examples))
        self.assertEqual(n_examples, samples, "NOT PASSED")


if __name__ == "__main__":
    unittest.main()
