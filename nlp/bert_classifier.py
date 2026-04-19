from transformers import pipeline

classifier = pipeline("text-classification")

def check(sentence):
    return classifier(sentence)