import pandas as pd


def create(filename, word_used_at_least_num_times=None, encoding=None):
    if word_used_at_least_num_times is None:
        word_used_at_least_num_times = 2

    if encoding is None:
        encoding = "utf8"

    words = set()
    frequencies = {}

    with open(filename, encoding=encoding) as input_file:
        for line in input_file:
            for word in line.split():
                word = word.lower()
                if word not in frequencies:
                    frequencies[word] = 1
                else:
                    frequencies[word] += 1

                if frequencies[word] >= word_used_at_least_num_times:
                    words.add(word)

    items = []
    index = 0
    for word in words:
        items.append((index,  [index, word]))
        index += 1

    return pd.DataFrame.from_items(items, columns=["index", "word"], orient="index")


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
