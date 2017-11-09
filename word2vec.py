import numpy
import pandas as pd
from keras import Input, callbacks
from keras.engine import Model
from keras.layers import Lambda, K, Dense
from keras.metrics import top_k_categorical_accuracy
from os.path import exists
import json
from helper_algorithms.words_kmeans import get_kmeans
from keras_models.models import get_w2v_model

USER = "macbosse"
TRAIN_META_CSV = "./train_data/"+USER+"_meta.csv"
TRAIN_VOCAB_CSV = "./train_data/"+USER+"_vocab.csv"
WORD2VEC_TRAIN = "./train_data/"+USER+"_w2v.csv"
WORD2VEC_TRAIN_MODEL = "./models/"+USER+"_w2v.model"
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

    if not exists(WORD2VEC_TRAIN_MODEL):
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

        encoding_layer = Dense(100, activation="linear", name="encoding_layer")(w1_input_hot)

        prediction = Dense(vec_len, activation="softmax")(encoding_layer)

        model = Model(inputs=[w1, w2], outputs=prediction)

        model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["categorical_accuracy", 'binary_accuracy'],
                      target_tensors=[w2_input_hot])

        h = model.fit(x=[trainings_set["word"], trainings_set["context_word"]], verbose=1,
                      epochs=50, batch_size=1000,
                      callbacks=[
                          callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True),
                          callbacks.ModelCheckpoint(WORD2VEC_TRAIN_MODEL, monitor='loss', verbose=0, save_best_only=True,save_weights_only=True, mode='auto', period=5),
                          #callbacks.EarlyStopping(monitor='loss', patience=2, verbose=0, mode='auto', min_delta=0.01)
                      ]
                      )

    ########################################################################################

    loaded_model = get_w2v_model(vec_len)
    loaded_model.load_weights(WORD2VEC_TRAIN_MODEL, by_name=True)

    p = loaded_model.predict(x=numpy.array([vocab_list[vocab_list["word"] == "great"]["index"].values[0]]))

    distances = []

    for word in vocab_list["word"]:
        p_w = loaded_model.predict(x=numpy.array([vocab_list[vocab_list["word"] == word]["index"].values[0]]))
        dist = numpy.linalg.norm(p.flatten().reshape((100,1)) - p_w.flatten().reshape((100,1)))
        #print("Obamacare -> "+word+" distance: "+str(dist))
        distances.append(("great", word, dist))
    sorted_distances = sorted(distances, key=lambda w: w[2])
    print(sorted_distances[:20])

    ########################################################################################

    word_centers, _ = get_kmeans(vocab_list, loaded_model)
    print(json.dumps(word_centers, indent=4, sort_keys=True))






