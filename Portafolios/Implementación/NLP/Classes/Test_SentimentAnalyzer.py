"""
Author: David Rodriguez Fragoso
Class that tests the SentimentAnalyzer class
Creation date: 29/11/2022
Last updated: 29/11/2022
"""
import unittest
from SentimentAnalyzer import SentimentAnalyzer
import os


class Test(unittest.TestCase):

    def test_analyze(self):
        new_text = ["I love doing tests", "I hate apples"]
        test = ["POSITIVE", "NEGATIVE"]
        analyzer = SentimentAnalyzer(new_text)
        result = analyzer.analyze()
        self.assertEqual(test, result, "NOT PASSED")


if __name__ == "__main__":
    unittest.main()
