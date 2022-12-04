"""
Author: David Rodriguez Fragoso
Class that tests the Translator class
Creation date: 29/11/2022
Last updated: 29/11/2022
"""
import unittest
from Translator import Translator
import os


class Test(unittest.TestCase):

    def test_translator(self):
        print(f'{"TESTING TRANSLATOR":-^40}')
        test = ["Hi this is a test\n", "I like apples\n"]
        test2 = ["Hola, esto es una prueba", "Me gustan las manzanas"]
        translator = Translator(test)
        argos_translate = Translator.argos(translator)
        self.assertEqual(test2, argos_translate, "NOT PASSED")


if __name__ == "__main__":
    unittest.main()

