import pandas as pd


def load(filename):
    return pd.read_csv(filename)


def save(vocab, filename):
    vocab.to_csv(filename, index=False)


def word_index(vocab, word):
    try:
        return vocab[vocab["word"] == word]["index"].values[0]
    except IndexError:
        return None


def word(vocab, index):
    try:
        return vocab["word"][index]
    except IndexError:
        return None
