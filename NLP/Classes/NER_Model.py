"""
Author: David Rodriguez Fragoso
NER model class with methods to train and plot the model
Creation date: 25/11/2022
Last updated: 25/11/2022
"""

import matplotlib.pyplot as plt
from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.embeddings import WordEmbeddings, StackedEmbeddings, TokenEmbeddings
from typing import List
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer


class NERModel:
    def __init__(self, examples, epochs, learning_rate, train_data, test_data, dev_data, data_folder, columns,
                 tag_type):
        self.examples = examples
        self.epochs = epochs
        self.learning_rate = learning_rate
        # Build the corpus using the txt data
        self.corpus: Corpus = ColumnCorpus(data_folder, columns,
                                           train_file='train1.txt',
                                           test_file='test.txt',
                                           dev_file='dev.txt')
        self.tag_type = tag_type

        # Build the tag dictionary using the tag type
        self.tag_dictionary = self.corpus.make_label_dictionary(label_type=tag_type)
        # Build the embedding types list
        self.embedding_types: List[TokenEmbeddings] = [
            WordEmbeddings('glove'),
            # other embeddings
        ]

        # Set the embeddings
        self.embeddings: StackedEmbeddings = StackedEmbeddings(
            embeddings=self.embedding_types)

        # Build the sequence tagger
        self.tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                                     embeddings=self.embeddings,
                                                     tag_dictionary=self.tag_dictionary,
                                                     tag_type=tag_type,
                                                     use_crf=True,
                                                     allow_unk_predictions=True)
        self.trainer: ModelTrainer = ModelTrainer(self.tagger, self.corpus)

    def trainModel(self):
        """
            Method that trains the model

            Parameters:

            Returns:

        """
        self.trainer.train('resources/taggers/example-ner',
                           learning_rate=self.learning_rate,
                           mini_batch_size=32,
                           max_epochs=self.epochs)

    @staticmethod
    def plot(data):
        """
            Method that plots the obtained results

            Parameters:
            * data: a pandas array with the loss data

            Returns:
            * Plots the results
        """
        plt.plot(data['EPOCH'], data['TRAIN_LOSS'], 'b-.')
        plt.plot(data['EPOCH'], data['DEV_LOSS'], 'r--')
        return print(plt.show())
