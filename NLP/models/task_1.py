'''
Author: David Rodriguez Fragoso
Script that tests a sentiment analisis model using a given dataset
Creation date: 07/11/2022
Last updated: 07/11/2022
'''

from transformers import pipeline

# open the .txt and append each line into a list
text = open("tiny_movie_reviews_dataset.txt","r")
input = text.readlines()

# create an empty list to save our preprocessed text
process = []

# remove the punctuation signs from the text 
for i in input:
  i = i.replace("#","")
  i = i.replace(".","")
  i = i.replace("!","")
  i = i.replace('"',"")
  i = i.replace("$","")
  i = i.replace("%","")
  i = i.replace("&","")
  i = i.replace("'","")
  i = i.replace("()","")
  i = i.replace("*","")
  i = i.replace("+","")
  i = i.replace(",","")
  i = i.replace("-","")
  i = i.replace("/","")
  i = i.replace(":","")
  i = i.replace(";","")
  i = i.replace("<","")
  i = i.replace("=","")
  i = i.replace(">","")
  i = i.replace("?","")
  i = i.replace("@","")
  i = i.replace("[","")
  i = i.replace("\\","")
  i = i.replace("]","")
  i = i.replace("^","")
  i = i.replace("_","")
  i = i.replace("`","")
  i = i.replace("{","")
  i = i.replace("}","")
  i = i.replace("|","")
  i = i.replace("~","")
  process.append(i)

# analyze our preprocessed text and print the result
def analyze(process):
    sentiment_pipeline = pipeline("sentiment-analysis")
    for i in sentiment_pipeline(process):
      print(i['label'])

analyze(process)
