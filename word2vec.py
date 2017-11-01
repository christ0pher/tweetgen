import pandas as pd
from keras import Input, callbacks
from keras.engine import Model
from keras.layers import Lambda, K, Dense, Maximum
from keras.metrics import top_k_categorical_accuracy
from os.path import exists

__author__ = 'christopher@levire.com'

USER = "realDonaldTrump"
TRAIN_META_CSV = "./train_data/"+USER+"_meta.csv"
TRAIN_VOCAB_CSV = "./train_data/"+USER+"_vocab.csv"
WORD2VEC_TRAIN = "./train_data/"+USER+"_w2v.csv"
WORD2VEC_TRAIN_MODEL = "D:\\models\\"+USER+"_w2v.model"
RAW_TEXT = "./raw_data/"+USER+".txt"


def top_3_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)


def wtoint(w, vocab):
    if len(vocab[vocab["word"] == w]["index"]) == 1:
        return vocab[vocab["word"] == w]["index"].values[0]
    return None


if __name__ == "__main__":

    meta_data = pd.read_csv(
        TRAIN_META_CSV
    )

    vocab_list = pd.read_csv(TRAIN_VOCAB_CSV)

    if not exists(WORD2VEC_TRAIN):
        with open(RAW_TEXT, encoding="utf-8") as input_file:
            with open(WORD2VEC_TRAIN, "w+") as w2vf:
                w2vf.write("word,context_word\n")
                for line in input_file:
                    words = line.split()
                    for i in range(0, len(words)):
                        w = wtoint(words[i], vocab_list)
                        if w is None:
                            continue
                        left_words = words[max(0, i-4):max(0, i-1)]
                        right_words = words[min(i+1, len(words)-1):min(i+4, len(words)-1)]
                        for context_word in left_words+right_words:
                            cw = wtoint(context_word, vocab_list)
                            if w is not None and cw is not None:
                                w2vf.write(str(w)+","+str(cw)+"\n")

    trainings_set = pd.read_csv(WORD2VEC_TRAIN)

    vec_len = meta_data["word_vec_len"][0]

    w1 = Input(shape=(1,), dtype="int32")
    w2 = Input(shape=(1,), dtype="int32")

    w1_input_hot = Lambda(
        K.one_hot,
        arguments={"num_classes": vec_len},
        output_shape=(1, vec_len),
        name="w1_input_hot"
    )(w1)

    w2_input_hot = Lambda(
        K.one_hot,
        arguments={"num_classes": vec_len},
        output_shape=(1, vec_len),
        name="w2_input_hot"
    )(w2)

    encoding_layer = Dense(100, activation="linear")(w1_input_hot)

    prediction = Dense(vec_len, activation="softmax")(encoding_layer)

    model = Model(inputs=[w1, w2], outputs=prediction)

    model.compile(optimizer="sgd", loss="categorical_crossentropy", metrics=["categorical_accuracy", 'binary_accuracy'],
                  target_tensors=[w2_input_hot])

    h = model.fit(x=[trainings_set["word"], trainings_set["context_word"]], verbose=1,
                  epochs=5000, batch_size=1000,
                  callbacks=[
                      callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True),
                      callbacks.ModelCheckpoint(WORD2VEC_TRAIN_MODEL, monitor='loss', verbose=0, save_best_only=True,save_weights_only=True, mode='auto', period=5),
                      #callbacks.EarlyStopping(monitor='loss', patience=2, verbose=0, mode='auto', min_delta=0.01)
                  ]
                  )
