"""
Author: David Rodriguez Fragoso
Script that tests a sentiment analysis model using a given dataset
Creation date: 07/11/2022
Last updated: 25/11/2022
"""
import string

from Classes.SentimentAnalyzer import SentimentAnalyzer

# open the .txt and save each example in separate lines
text = open("../tiny_movie_reviews_dataset.txt", "r")
text = text.readlines()

# remove the punctuation signs from the text and save it
new_text = [character.translate(str.maketrans('', '', string.punctuation)) for character in text]

# Create the analyzer instance
analyzer = SentimentAnalyzer(new_text)

# Call the analyze method
analyzer.analyze()


'''if __name__ == '__main__':
    print("--------------------TESTS--------------------")
    process = ["I love doing tests", "I hate apples"]
    analyze(process)
'''