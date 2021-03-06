import vocabulary
import sys
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import load_model


def generate_next_word(word1, word2, word3, model, vocab, head=5):
    w1 = np.array([vocabulary.word_index(vocab, word1)])
    w2 = np.array([vocabulary.word_index(vocab, word2)])
    w3 = np.array([vocabulary.word_index(vocab, word3)])
    prediction = model.predict(x=[w1, w2, w3]).flatten()

    indices = np.argsort(-prediction)[:head]
    return vocabulary.word(vocab, index=indices[random.randint(0, head - 1)])


if __name__ == "__main__":
    user = sys.argv[1]
    words = [sys.argv[2], sys.argv[3], sys.argv[4]]
    head = int(sys.argv[5])
    tweets = int(sys.argv[6])

    TRAIN_VOCAB_CSV = "./train_data/" + user + "_vocab.csv"
    MODEL_FILE = "./models/" + user + ".model"

    vocab_list = pd.read_csv(TRAIN_VOCAB_CSV, engine="python", sep=",")

    model = load_model(MODEL_FILE, custom_objects={"tf": tf})

    for tweet in range(tweets):
        while True:
            next_word = generate_next_word(words[-3], words[-2], words[-1], model, vocab_list, head=head)
            words.append(next_word)
            if len(" ".join(words)) >= 140:
                words.pop()
                break

        print(" ".join(words))
        words = [sys.argv[2], sys.argv[3], sys.argv[4]]
