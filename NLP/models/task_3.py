'''
Author: David Rodriguez Fragoso
Script that trains an NLP model using a given dataset
Creation date: 10/11/2022
Last updated: 11/11/2022
'''

from nltk.translate.bleu_score import sentence_bleu
from Classes.Translate import Translate

# open the text to be translated
text = open("../europarl-v7.es-en.en", "r",encoding="utf-8")
text = text.readlines()

# open test dataset
text2 = open("../europarl-v7.es-en.es", "r",encoding="utf-8")
text2 = text2.readlines()
'''
# argos api function
def argos_translate(text):

    for example in text:
        # body for argos api request
        body = {
            "q": example,
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
    for example in text:
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
        req2 = requests.request("POST", url2, json=payload, headers=headers)
        res2 = json.loads(req2.text)
        dt_api.append(res2['data']['translations']['translatedText'])

    return dt_api

'''


#Save the translation results
argos_translate = Translate.argos_translate(text)
dt_translate = Translate.dt_translate(text)

# calculate average bleu for google api and the argos api
bleu_argos = Translate.obtainBLEU(argos_translate)
bleu_dt = Translate.obtainBLEU(dt_translate)


print(f"ARGOS_API: {bleu_argos}")
print(f"DeepTranslate_API: {bleu_dt}")

'''if __name__ == '__main__':
    print("--------------------TESTS--------------------")
    test = ["Hi this is a test", "I like apples"]
    test2 = ["Hola esto es un test", "Me gustan las manzanas"]
    argos_translate(test)
    dt_translate(test)
    bleu_argos = []
    bleu_dt = []

    # calculate average bleu for argos api
    for text in test2:
        bleu_argos.append(sentence_bleu(argos_api, i))
    bleu_argos = sum(bleu_argos)/len(bleu_argos)

    # calculate average bleu for google api
    for text in test2:
        bleu_dt.append(sentence_bleu(dt_api, i))
    bleu_google = sum(bleu_dt)/len(bleu_dt)


    print(f"ARGOS_API: {bleu_argos}")
    print(f"DeepTranslate_API: {bleu_google}")'''