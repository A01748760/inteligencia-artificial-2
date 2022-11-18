'''
Author: David Rodriguez Fragoso
Script that trains an NLP model using a given dataset
Creation date: 10/11/2022
Last updated: 11/11/2022
'''

"""

I would make this into a Translate class which has methods like the ones you have below -- it will help your code be clean and organized! 
Examples: https://www.dataquest.io/blog/using-classes-in-python/ and https://www.geeksforgeeks.org/python-classes-and-objects/



And be sure to add tests for all pieces of functionality! 

Tests should check that the functionality/outputs are as expected, not just that the code runs! 

When writing code, I write the test for each class or piece of functionality as soon as I finish that piece.
It helps you develop incrementally, being sure that each piece of code is clean and works like you expect it to! 


for tests, best practices are to have a structure like this: 
https://stackoverflow.com/questions/1896918/running-unittest-with-typical-test-directory-structure

"""


import json
import requests
from nltk.translate.bleu_score import sentence_bleu
import dotenv
import os

from dotenv import load_dotenv

load_dotenv()

# api url's
url1 = 'https://translate.argosopentech.com/translate'
url2 = 'https://deep-translate1.p.rapidapi.com/language/translate/v2'

# translation results
argos_api = []
dt_api = []

# bleu for both api's
bleu_argos = []
bleu_dt = []

# test text
esp = []

# open the text to be translated
text = open("europarl-v7.es-en.en", "r",encoding="utf-8")
text = text.readlines()

# argos api function
def argos_translate(text):

    for i in text:
        # body for argos api request
        body = {
            "q": i,
            "source": "en",
            "target": "es"
        }

        # argos api request
        req1 = requests.request("POST", url1, data=json.dumps(body), headers={"content-type": "application/json"})
        res1 = json.loads(req1.text)
        argos_api.append(res1["translatedText"][:-1])
    
    return argos_api

# deep translate api function
def dt_translate(text):
    for i in text:
        # body for google api request
        payload = {
            "q": i,
            "source": "en",
            "target": "es"
        }
        headers = {
            "content-type": "application/json",
            "X-RapidAPI-Key": os.getenv("X-RapidAPI-Key"),
            "X-RapidAPI-Host": "deep-translate1.p.rapidapi.com"
        }

        # google api request
        req2 = requests.request("POST", url2, json=payload, headers=headers)
        res2 = json.loads(req2.text)
        dt_api.append(res2['data']['translations']['translatedText'])

    return dt_api


# open test dataset
text2 = open("europarl-v7.es-en.es", "r",encoding="utf-8")
text2 = text2.readlines()

argos_translate(text)
dt_translate(text)

# calculate average bleu for argos api
for i in text2:  # name this text rather than i 
    bleu_argos.append(sentence_bleu(argos_api, i))
bleu_argos = sum(bleu_argos)/len(bleu_argos)

# calculate average bleu for google api
for i in text2:
    bleu_dt.append(sentence_bleu(dt_api, i))
bleu_google = sum(bleu_dt)/len(bleu_dt)


print(f"ARGOS_API: {bleu_argos}")
print(f"DeepTranslate_API: {bleu_google}")

if __name__ == '__main__':
    print("--------------------TESTS--------------------")
    test = ["Hi this is a test", "I like apples"]
    test2 = ["Hola esto es un test", "Me gustan las manzanas"]
    argos_translate(test)
    dt_translate(test)
    bleu_argos = []
    bleu_dt = []

    # calculate average bleu for argos api
    for i in test2:
        bleu_argos.append(sentence_bleu(argos_api, i))
    bleu_argos = sum(bleu_argos)/len(bleu_argos)

    # calculate average bleu for google api
    for i in test2:
        bleu_dt.append(sentence_bleu(dt_api, i))
    bleu_google = sum(bleu_dt)/len(bleu_dt)


    print(f"ARGOS_API: {bleu_argos}")
    print(f"DeepTranslate_API: {bleu_google}")
