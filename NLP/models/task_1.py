'''
Author: David Rodriguez Fragoso
Script that tests a sentiment analisis model using a given dataset
Creation date: 07/11/2022
Last updated: 07/11/2022
'''
"""
Alwasy put your code in a descriptive method or small class, if possible, rather than running at the top-level. This makes it easier for other modules to import the functionality later if needed! 


Also, tests should be in a separate directory! 
Best practices are to have a structure like this: https://stackoverflow.com/questions/1896918/running-unittest-with-typical-test-directory-structure

And tests should assert that the output is correct...not just check that the process runs! 
"""
from transformers import pipeline

# open the .txt and append each line into a list
text = open("tiny_movie_reviews_dataset.txt","r")
input = text.readlines()

# create an empty list to save our preprocessed text
process = []


# Instead of creating your own, just use a prebuilt way to remove punctuation: https://datagy.io/python-remove-punctuation-from-string/#:~:text=One%20of%20the%20easiest%20ways,maketrans()%20method.
# however, for this task, it might be better to leave punctuation! Since punctuation gives us some information about the sentiment. 
# 
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


if __name__ == '__main__':
  print("--------------------TESTS--------------------")
  process = ["I love doing tests", "I hate apples"]
  analyze(process)
