'''
Author: David Rodriguez Fragoso
Script that tests a sentiment analisis model using a given dataset
Creation date: 07/11/2022
Last updated: 07/11/2022
'''

from transformers import pipeline

# open the .txt and append each line into a list
text = open("../tiny_movie_reviews_dataset.txt", "r")
input = text.readlines()

# create an empty list to save our preprocessed text
process = []

# remove the punctuation signs from the text 
for character in input:
  character = character.replace("#","")
  character = character.replace(".","")
  character = character.replace("!","")
  character = character.replace('"',"")
  character = character.replace("$","")
  character = character.replace("%","")
  character = character.replace("&","")
  character = character.replace("'","")
  character = character.replace("()","")
  character = character.replace("*","")
  character = character.replace("+","")
  character = character.replace(",","")
  character = character.replace("-","")
  character = character.replace("/","")
  character = character.replace(":","")
  character = character.replace(";","")
  character = character.replace("<","")
  character = character.replace("=","")
  character = character.replace(">","")
  character = character.replace("?","")
  character = character.replace("@","")
  character = character.replace("[","")
  character = character.replace("\\","")
  character = character.replace("]","")
  character = character.replace("^","")
  character = character.replace("_","")
  character = character.replace("`","")
  character = character.replace("{","")
  character = character.replace("}","")
  character = character.replace("|","")
  character = character.replace("~","")
  process.append(character)

# analyze our preprocessed text and print the result
def analyze(process):
    sentiment_pipeline = pipeline("sentiment-analysis")
    for example in sentiment_pipeline(process):
      print(example['label'])

analyze(process)


if __name__ == '__main__':
  print("--------------------TESTS--------------------")
  process = ["I love doing tests", "I hate apples"]
  analyze(process)
