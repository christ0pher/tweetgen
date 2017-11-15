import numpy
import pandas as pd
import vocabulary
import filepath
import trainingsset
import model
import keras
from keras import Input
from keras.layers import Lambda, K, Dense
from keras.engine import Model
from sklearn.cluster import KMeans
import os
import json

USER = "macbosse"


def get_kmeans(vocab_list, w2v_model):
    vectors = []
    for word in vocab_list["word"]:
        vec = w2v_model.predict(x=numpy.array([vocab_list[vocab_list["word"] == word]["index"].values[0]]))
        vectors.append(vec.flatten())

    kmeans = KMeans(init="random", n_clusters=15)
    kmeans.fit(numpy.array(vectors))

    word_centers = {}
    clusters = {}
    for word in vocab_list["word"]:
        vec = w2v_model.predict(x=numpy.array([vocab_list[vocab_list["word"] == word]["index"].values[0]]))
        cluster = kmeans.predict(vec.flatten().reshape(1, -1))[0]
        h = str(cluster)
        if h not in word_centers:
            word_centers[h] = []
        word_centers[h].append(word)
        clusters[vocab_list[vocab_list["word"] == word]["index"].values[0]] = cluster
    return word_centers, clusters


if __name__ == "__main__":
    print("Loading vocabulary for %s." % USER)
    vocab = vocabulary.load(filepath.for_vocabulary_file(USER))

    if not os.path.exists(filepath.for_skip_gram_file(USER)):
        print("No skip gram trainings set exists for %s. Generating new trainings set ..." % USER)
        trainings_set = trainingsset.generate_skip_gram(filepath.for_raw_text_file(USER), vocab, 3)
        trainings_set.to_csv(filepath.for_skip_gram_file(USER), index=False)
        print("DONE")
    else:
        print("Loading skip gram trainings set for %s." % USER)
        trainings_set = pd.read_csv(filepath.for_skip_gram_file(USER))

    vocab_len = len(vocab)
    w2v_model_path = filepath.for_w2v_model(USER)

    if not os.path.exists(w2v_model_path):
        print("No pre trained word2vec model found for %s. Training new word2vec model ..." % USER)

        word_input, encoding = model.word2vec_embedding(vocab_len, vec_size=100)

        target_word_input = Input(shape=(1,), dtype="int32")
        target_word_hot = Lambda(
            K.one_hot,
            arguments={"num_classes": vocab_len},
            output_shape=(1, vocab_len),
            name="w1_input_hot"
        )(target_word_input)

        prediction = Dense(vocab_len, activation="softmax")(encoding)

        w2v_model = Model(inputs=[word_input, target_word_input], outputs=prediction)

        w2v_model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["categorical_accuracy"],
            target_tensors=[target_word_hot]
        )

        w2v_model.fit(
            x=[trainings_set["word"], trainings_set["context"]],
            verbose=1,
            epochs=50,
            batch_size=1000,
            callbacks=[
                keras.callbacks.TensorBoard(
                    log_dir='./Graph',
                    histogram_freq=0,
                    write_graph=True,
                    write_images=True
                ),
                keras.callbacks.ModelCheckpoint(
                    w2v_model_path,
                    monitor='loss',
                    verbose=0,
                    save_best_only=True,
                    save_weights_only=True,
                    mode='auto',
                    period=5
                ),
                keras.callbacks.EarlyStopping(
                    monitor='loss',
                    patience=2,
                    verbose=0,
                    mode='auto',
                    min_delta=0.01
                )
            ]
        )

    w2v_model = model.word2vec(vocab_len, vec_size=100)
    w2v_model.load_weights(w2v_model_path, by_name=True)

    p = w2v_model.predict(x=numpy.array([vocabulary.word_index(vocab, "great")]))

    distances = []

    for word in vocab["word"]:
        p_w = w2v_model.predict(x=numpy.array([vocabulary.word_index(vocab, word)]))
        dist = numpy.linalg.norm(p.flatten().reshape((100, 1)) - p_w.flatten().reshape((100, 1)))
        distances.append(("great", word, dist))
    sorted_distances = sorted(distances, key=lambda w: w[2])
    print(sorted_distances[:20])

    word_centers, _ = get_kmeans(vocab, w2v_model)
    print(json.dumps(word_centers, indent=4, sort_keys=True))






