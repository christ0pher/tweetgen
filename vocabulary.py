import pandas as pd


def load(filename):
    return pd.read_csv(filename)


def word_index(vocab, word):
    return vocab[vocab["word"] == word]["index"].values[0]


def word(vocab, index):
    return vocab["word"][index]
