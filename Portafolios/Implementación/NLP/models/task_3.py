"""
Author: David Rodriguez Fragoso
Script that trains an NLP model using a given dataset
Creation date: 10/11/2022
Last updated: 11/11/2022
"""

from Classes.Translator import Translator

# open the text to be translated
text = open("../europarl-v7.es-en.en", "r", encoding="utf-8")
text = text.readlines()

# open test dataset
text2 = open("../europarl-v7.es-en.es", "r", encoding="utf-8")
text2 = text2.readlines()

# Create the Translate class instance
translator = Translator(text)

# Save the translated text
argos_translate = Translator.argos(translator)
dt_translate = Translator.deepTranslate(translator)

# calculate average bleu for google api and the argos api
bleu_argos = Translator.obtainBLEU(argos_translate, text2)
bleu_dt = Translator.obtainBLEU(dt_translate, text2)


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