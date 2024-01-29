from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import numpy as np
import torch

MODELS = {
    'distilbert': 'distilbert-base-uncased-finetuned-sst-2-english',
    'xlm-roberta': 'cardiffnlp/twitter-xlm-roberta-base-sentiment'
}

LABELS = {
    0: "NEGATIVE",
    1: "POSITIVE"
}

PRECISION = 6


class Classify:
    def __init__(self, model_name):
        self.model_name = MODELS[model_name]

    def evaluate(self, text):
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        model_input = tokenizer(text, return_tensors='pt')
        logits = None
        with torch.no_grad():
            logits = model(**model_input).logits

        if logits is not None:
            logits_d = logits.detach().numpy()
            probabilities = softmax(logits_d)
            prediction = np.argmax(probabilities, axis=1)
            print(probabilities)

            proba = "{ Neg: " + str(round(probabilities[0][0], PRECISION)) + ", " + " Pos: " + str(
                round(probabilities[0][1], PRECISION)) + "\n}"
            classification = LABELS[prediction[0]]

            return self.model_name, text, proba, classification

