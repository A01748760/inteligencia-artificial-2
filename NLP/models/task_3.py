import json
import requests
from nltk.translate.bleu_score import sentence_bleu

# api url's
url1 = 'https://translate.argosopentech.com/translate'
url2 = 'https://google-translate1.p.rapidapi.com/language/translate/v2'

# translation results
argos_api = []
google_api = []

# bleu for both api's
bleu_argos = []
bleu_google = []

# test text
esp = []

# open the text to be translated
text = open("europarl1-v7.es-en.en", "r",encoding="utf-8")
text = text.readlines()

# iterate over the text to be translated
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
    
    # body for google api request
    payload = f"q={i}&target=es&source=en"
    headers = {
        "content-type": "application/x-www-form-urlencoded",
        "Accept-Encoding": "application/gzip",
        "X-RapidAPI-Key": "b35c735f7emshcc41b1bb3beabc8p1360fbjsn81baa316bc3e",
        "X-RapidAPI-Host": "google-translate1.p.rapidapi.com"
    }

    # google api request
    req2 = requests.request("POST", url2, data=payload, headers=headers)
    res2 = json.loads(req2.text)
    print(res2)
    google_api.append(res2["data"]["translations"][0]["translatedText"])


# open test dataset
text2 = open("europarl1-v7.es-en.es", "r",encoding="utf-8")
text2 = text2.readlines()

# calculate average bleu for google api
for i in text2:
    bleu_google.append(sentence_bleu(google_api, i))
bleu_google = sum(bleu_google)/len(bleu_google)

# calculate average bleu for argos api
for i in text2:
    bleu_argos.append(sentence_bleu(argos_api, i))
bleu_argos = sum(bleu_argos)/len(bleu_argos)

# print average BLEU 
print(f"ARGOS_API: {bleu_argos}")
print(f"GOOGLE_API: {bleu_google}")
