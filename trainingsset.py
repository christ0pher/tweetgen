import vocabulary
import pandas as pd


def generate_ngram_to_next_word(text_file, vocab, n=None, encoding=None):
    n = 3 if n is None else n
    encoding = "utf8" if encoding is None else encoding

    with open(text_file, encoding=encoding) as tf:
        entries = []
        for line in tf:
            words = line.split()
            for i in range(len(words) - n):
                entry = []
                for j in range(n):
                    index = vocabulary.word_index(vocab, words[i+j])
                    if index is not None:
                        entry.append(index)

                index = vocabulary.word_index(vocab, words[i + n])
                if index is not None:
                    entry.append(index)

                if len(entry) == n+1:
                    entries.append(entry)

    columns = ["w"+str(i+1) for i in range(n)]
    columns.append("target")

    return pd.DataFrame(entries, columns=columns)
