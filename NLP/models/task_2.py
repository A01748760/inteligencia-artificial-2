'''
Author: David Rodriguez Fragoso
Script that trains an NLP model using a given dataset
Creation date: 08/11/2022
Last updated: 08/11/2022
'''

"""

It’s best practice to have your code in a descriptive method or small class, if possible, rather than running at the top-level. 
This makes it easier for other modules to import the functionality later if needed! 


Try to split your code into methods and classes with descriptive names! Some reading for this: 

Classes: https://www.dataquest.io/blog/using-classes-in-python/ and https://www.geeksforgeeks.org/python-classes-and-objects/
# Classes: https://www.dataquest.io/blog/using-classes-in-python/
"""

from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.embeddings import WordEmbeddings, StackedEmbeddings, TokenEmbeddings
from typing import List
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
import pandas as pd
import matplotlib.pyplot as plt


N_EXAMPLES_TO_TRAIN = 500
EPOCHES = 10
LEARNING_RATE = 0.05

train = []

with open("../train.txt") as myfile:
        for N in range(N_EXAMPLES_TO_TRAIN):
                train.append(next(myfile))

# create the new test file
test_file = open("../train1.txt", "w")
test_file.writelines(train)
test_file.close()

# define columns
columns = {0 : 'text', 1 : 'ner'}
# directory where the data resides
data_folder = '../'
# initializing the corpus


corpus: Corpus = ColumnCorpus(data_folder, columns,
                              train_file = 'train1.txt',
                              test_file = 'test.txt',
                              dev_file = 'dev.txt')

# tag to predict
tag_type = 'ner'
# make tag dictionary from the corpus
tag_dictionary = corpus.make_label_dictionary(label_type=tag_type)


embedding_types : List[TokenEmbeddings] = [
        WordEmbeddings('glove'),
        ## other embeddings
        ]

embeddings : StackedEmbeddings = StackedEmbeddings(
                                 embeddings=embedding_types)

tagger : SequenceTagger = SequenceTagger(hidden_size=256,
                                       embeddings=embeddings,
                                       tag_dictionary=tag_dictionary,
                                       tag_type=tag_type,
                                       use_crf=True,
                                       allow_unk_predictions = True)

trainer : ModelTrainer = ModelTrainer(tagger, corpus)
trainer.train('resources/taggers/example-ner',
              learning_rate=LEARNING_RATE,
              mini_batch_size=32,
              max_epochs=EPOCHES)

def plot():
        loss_data = pd.read_csv('resources/taggers/example-ner/loss.tsv', sep='\t')

        plt.plot(loss_data["EPOCH"],loss_data["TRAIN_LOSS"],'b-.')
        plt.plot(loss_data["EPOCH"],loss_data["DEV_LOSS"],'r--')
        plt.show()
