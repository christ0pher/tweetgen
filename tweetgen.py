import pandas as pd
from keras import Input
from keras.engine import Model
from keras.layers import Lambda, Dense, Concatenate
import keras.backend as K

USER = "khloekardashian"
TRAIN_CSV = "./train_data/"+USER+".csv"
TRAIN_META_CSV = "./train_data/"+USER+"_meta.csv"
TRAIN_VOCAB_CSV = "./train_data/"+USER+"_vocab.csv"
MODEL_FILE = "./models/" + USER + ".model"

if __name__ == "__main__":

    meta_data = pd.read_csv(
        TRAIN_META_CSV
    )

    trainings_set = pd.read_csv(TRAIN_CSV)

    vocab_list = pd.read_csv(TRAIN_VOCAB_CSV)

    vec_len = meta_data["word_vec_len"][0]

    print(vec_len)

    w1_input = Input(shape=(1,), dtype="int32")
    w2_input = Input(shape=(1,), dtype="int32")
    w3_input = Input(shape=(1,), dtype="int32")
    target_input = Input(shape=(1,), dtype="int32")

    w1_one_hot = Lambda(
        K.one_hot,
        arguments={"num_classes": vec_len},
        output_shape=(1, vec_len),
        name="w1_one_hot"
    )(w1_input)

    w2_one_hot = Lambda(
        K.one_hot,
        arguments={"num_classes": vec_len},
        output_shape=(1, vec_len),
        name="w2_one_hot"
    )(w2_input)

    w3_one_hot = Lambda(
        K.one_hot,
        arguments={"num_classes": vec_len},
        output_shape=(1, vec_len),
        name="w3_one_hot"
    )(w3_input)

    target_one_hot = Lambda(
        K.one_hot,
        arguments={"num_classes": vec_len},
        output_shape=(1, vec_len),
        name="target_one_hot"
    )(target_input)

    word_embedding = Dense(1024, activation="linear")

    w1_embedding = word_embedding(w1_one_hot)
    w2_embedding = word_embedding(w2_one_hot)
    w3_embedding = word_embedding(w3_one_hot)

    embedding = Concatenate()([w1_embedding, w2_embedding, w3_embedding])

    hidden = Dense(512, activation="sigmoid")(embedding)

    hidden2 = Dense(128, activation="hard_sigmoid")(hidden)
    prediction_layer = Dense(vec_len, activation="softmax")(hidden2)

    model = Model(inputs=[w1_input, w2_input, w3_input, target_input], outputs=prediction_layer)

    model.compile(
        optimizer="rmsprop",
        loss="categorical_crossentropy",
        metrics=["categorical_accuracy"],
        target_tensors=[target_one_hot]
    )

    print(model.summary())

    h = model.fit(
        x=[trainings_set["w1"],
           trainings_set["w2"],
           trainings_set["w3"],
           trainings_set["target"]
           ],
        verbose=1,
        epochs=200,
        batch_size=500
    )

    model.save(MODEL_FILE)

    print(h)
