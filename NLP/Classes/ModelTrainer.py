import spacy
from spacy import displacy


class ModelTrainer:

    def __init__(self, dataset):
        self.dataset = dataset
    

    def load_dataset(self):
        dataset = spacy.load("en_core_web_sm")
        return dataset
    
