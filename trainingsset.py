import vocabulary
import pandas as pd


def generate_n_gram_to_next_word(text_file, vocab, n=None, encoding=None):
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


def generate_skip_gram(text_file, vocab, n=None, encoding=None):
    n = 3 if n is None else n
    encoding = "utf8" if encoding is None else encoding

    with open(text_file, encoding=encoding) as tf:
        entries = []
        for line in tf:
            words = line.split()
            for i in range(len(words)):
                index = vocabulary.word_index(vocab, words[i])
                if index is None:
                    continue
                left_words = words[max(0, i-n):max(0, i)]
                right_words = words[min(i+1, len(words)-1):min(i+n+1, len(words))]
                for context_word in left_words+right_words:
                    context_word_index = vocabulary.word_index(vocab, context_word)
                    if index is not None and context_word_index is not None:
                        entries.append([index, context_word_index])

    return pd.DataFrame(entries, columns=["word", "context"])
