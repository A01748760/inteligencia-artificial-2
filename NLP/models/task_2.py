from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.embeddings import WordEmbeddings, StackedEmbeddings, TokenEmbeddings
from typing import List
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
import pandas as pd
import matplotlib.pyplot as plt


N_EXAMPLES_TO_TRAIN = 500
EPOCHES = 15
LEARNING_RATE = 0.05

train = []

with open("train.txt") as myfile:
        for x in range(N_EXAMPLES_TO_TRAIN):
                train.append(next(myfile))

# create the new test file
f = open("train1.txt", "w")
f.writelines(train)
f.close()

# define columns
columns = {0 : 'text', 1 : 'ner'}
# directory where the data resides
data_folder = './'
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
print(tagger)

loss_data = pd.read_csv('resources/taggers/example-ner/loss.tsv', sep='\t')
print(loss_data)
plt.plot(loss_data["EPOCH"],loss_data["TRAIN_LOSS"],'b-.')
plt.plot(loss_data["EPOCH"],loss_data["DEV_LOSS"],'r--')
plt.show()
