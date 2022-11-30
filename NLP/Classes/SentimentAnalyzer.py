"""
Author: David Rodriguez Fragoso
Class that implements methods for testing the sentiment analysis model
Creation date: 07/11/2022
Last updated: 25/11/2022
"""
from transformers import pipeline


class SentimentAnalyzer:

    def __init__(self, text):
        self.text = text

    def analyze(self):
        """
            Method that analyzes a given text

            Parameters:
            * Text: text to be analyzed

            Returns:
            * Prints the detected emotions in the console

        """
        sentiments = []
        sentiment_pipeline = pipeline("sentiment-analysis")
        for example in sentiment_pipeline(self.text):
            print(example['label'])
            sentiments.append(example['label'])

        return sentiments
