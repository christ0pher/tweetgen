import numpy
import pandas as pd
import vocabulary
import filepath
import trainingsset
from keras import Input, callbacks
from keras.engine import Model
from keras.layers import Lambda, K, Dense
from keras.metrics import top_k_categorical_accuracy
from os.path import exists
import json
from helper_algorithms.words_kmeans import get_kmeans
from keras_models.models import get_w2v_model

USER = "macbosse"

def top_3_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)


if __name__ == "__main__":
    print("Loading vocabulary for %s." % USER)
    vocab = vocabulary.load(filepath.for_vocabulary_file(USER))

    if not exists(filepath.for_skip_gram_file(USER)):
        print("No skip gram trainings set exists for %s. Generating new trainings set ..." % USER)
        trainings_set = trainingsset.generate_skip_gram(filepath.for_raw_text_file(USER), vocab, 3)
        trainings_set.to_csv(filepath.for_skip_gram_file(USER), index=False)
        print("DONE")
    else:
        print("Loading skip gram trainings set for %s." % USER)
        trainings_set = pd.read_csv(filepath.for_skip_gram_file(USER))

    vec_len = len(vocab)

    w2v_model_path = filepath.for_w2v_model(USER)
    if not exists(w2v_model_path):
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

        h = model.fit(x=[trainings_set["word"], trainings_set["context"]], verbose=1,
                      epochs=50, batch_size=1000,
                      callbacks=[
                          callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True),
                          callbacks.ModelCheckpoint(w2v_model_path, monitor='loss', verbose=0, save_best_only=True,save_weights_only=True, mode='auto', period=5),
                          #callbacks.EarlyStopping(monitor='loss', patience=2, verbose=0, mode='auto', min_delta=0.01)
                      ]
                      )

    loaded_model = get_w2v_model(vec_len)
    loaded_model.load_weights(w2v_model_path, by_name=True)

    p = loaded_model.predict(x=numpy.array([vocabulary.word_index(vocab, "great")]))

    distances = []

    for word in vocab["word"]:
        p_w = loaded_model.predict(x=numpy.array([vocabulary.word_index(vocab, word)]))
        dist = numpy.linalg.norm(p.flatten().reshape((100,1)) - p_w.flatten().reshape((100, 1)))
        distances.append(("great", word, dist))
    sorted_distances = sorted(distances, key=lambda w: w[2])
    print(sorted_distances[:20])

    word_centers, _ = get_kmeans(vocab, loaded_model)
    print(json.dumps(word_centers, indent=4, sort_keys=True))






