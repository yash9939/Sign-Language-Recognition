# ngram_model.py
from collections import defaultdict, Counter
import re

class NGramModel:
    def __init__(self, n=2):
        self.n = n
        self.model = defaultdict(Counter)
        self.vocab = set()

    def train(self, corpus):
        corpus = corpus.upper()
        words = re.findall(r"[A-Z]+", corpus)

        for word in words:
            self.vocab.add(word)
            padded = "_" * (self.n - 1) + word
            for i in range(len(word)):
                context = padded[i:i+self.n-1]
                next_char = word[i]
                self.model[context][next_char] += 1

    def predict_next(self, context):
        context = context[-(self.n-1):]
        if context not in self.model:
            return []
        counts = self.model[context]
        total = sum(counts.values())
        probs = {c: counts[c] / total for c in counts}
        return sorted(probs.items(), key=lambda x: x[1], reverse=True)

    def suggest_words(self, prefix, top_k=5):
        prefix = prefix.upper()
        if len(prefix) == 0:
            return []

        suggestions = []

        # Filter known words by prefix
        for word in self.vocab:
            if word.startswith(prefix):
                score = 0
                padded = "_" * (self.n - 1) + word
                for i in range(len(word)):
                    context = padded[i:i+self.n-1]
                    char = word[i]
                    if context in self.model and char in self.model[context]:
                        score += self.model[context][char]
                suggestions.append((word, score))

        suggestions.sort(key=lambda x: x[1], reverse=True)
        return [w for w, _ in suggestions[:top_k]]