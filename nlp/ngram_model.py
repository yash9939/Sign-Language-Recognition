import nltk
from collections import Counter

def build_ngram(words, n=2):
    ngrams = []
    for word in words:
        ngrams.extend(list(nltk.ngrams(word, n)))
    return Counter(ngrams)