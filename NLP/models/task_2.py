"""
Author: David Rodriguez Fragoso
Script that trains an NER model using a given dataset
Creation date: 08/11/2022
Last updated: 25/11/2022
"""

import pandas as pd
from Classes.NER_Model import NERModel

# Hyperparameter settings
N_EXAMPLES_TO_TRAIN = 500
EPOCHS = 7
LEARNING_RATE = 0.05

# Files settings
train_file = 'train.txt'
test_file = 'test.txt'
dev_file = 'dev.txt'

# Directory where the data resides
data_folder = '../'


train = NERModel.downsample(N_EXAMPLES_TO_TRAIN)
print(len(train))
# Create the new test file
test_file = open("../train1.txt", "w")
test_file.writelines(train)
test_file.close()

# Define columns
columns = {0: 'text', 1: 'ner'}

# Tag to predict
tag_type = 'ner'

# Initializing the model
ner_model = NERModel(N_EXAMPLES_TO_TRAIN, EPOCHS, LEARNING_RATE, train_file, test_file, dev_file, data_folder, columns,
                     tag_type)

# Train the model
trained_model = ner_model.trainModel()

# Save the loss data
loss_data = pd.read_csv('resources/taggers/example-ner/loss.tsv', sep='\t')

# Plot the results (test+train)
ner_model.plot(loss_data)

