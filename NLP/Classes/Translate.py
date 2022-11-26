"""
Author: David Rodriguez Fragoso
Class with methods to translate text using apis and obtain the BLEU
Creation date: 24/11/2022
Last updated: 24/11/2022
"""
import json
import requests
from nltk.translate.bleu_score import sentence_bleu

import os


class Translate:

    def __init__(self, text) -> None:
        self.text = text
    
    def argos(self):
        """
            Method that translates text using the argos api

            Parameters:
            * text: text to be translated

            Returns:
            * An array with the translated text
        """
        # api url
        url = 'https://translate.argosopentech.com/translate'
        # translation result
        argos_api = []
        for example in self.text:
            # body for argos api request
            body = {
                "q": example,
                "source": "en",
                "target": "es"
            }

            # argos api request
            req1 = requests.request("POST", url, data=json.dumps(body), headers={"content-type": "application/json"})
            res1 = json.loads(req1.text)
            argos_api.append(res1["translatedText"][:-1])
        return argos_api

    def deepTranslate(self):
        """
            Method that translates text using the deepTranslate api

            Parameters:
            * text: text to be translated

            Returns:
            * An array with the translated text
        """
        url = 'https://deep-translate1.p.rapidapi.com/language/translate/v2'
        # translation result
        dt_api = []

        for example in self.text:
            # body for google api request
            payload = {
                "q": example,
                "source": "en",
                "target": "es"
            }
            headers = {
                "content-type": "application/json",
                "X-RapidAPI-Key": os.getenv("X-RapidAPI-Key"),
                "X-RapidAPI-Host": "deep-translate1.p.rapidapi.com"
            }

            # google api request
            req2 = requests.request("POST", url, json=payload, headers=headers)
            res2 = json.loads(req2.text)
            dt_api.append(res2['data']['translations']['translatedText'])

            return dt_api

    @staticmethod
    def obtainBLEU(translation, test):
        """
            Class method that translates text using the deepTranslate api

            Parameters:
            * text: text to be translated

            Returns:
            * An array with the translated text
        """
        bleu = []
        for example in test:
            bleu.append(sentence_bleu(translation, example))
        
        return sum(bleu)/len(bleu)
