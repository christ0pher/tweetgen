import pandas as pd
from keras import Input
from keras.engine import Model
from keras.layers import Lambda, K, Concatenate, Dense, Maximum
from keras.metrics import top_k_categorical_accuracy

__author__ = 'christopher@levire.com'

USER = "khloekardashian"
TRAIN_META_CSV = "./train_data/"+USER+"_meta.csv"
TRAIN_VOCAB_CSV = "./train_data/"+USER+"_vocab.csv"
TRAIN_CSV = "./train_data/"+USER+".csv"


def top_3_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)


if __name__ == "__main__":

    meta_data = pd.read_csv(
        TRAIN_META_CSV
    )
    trainings_set = pd.read_csv(TRAIN_CSV)

    vec_len = meta_data["word_vec_len"][0]

    w1_target = Input(shape=(1,), dtype="int32")
    w2_target = Input(shape=(1,), dtype="int32")
    w3_target = Input(shape=(1,), dtype="int32")
    w_input = Input(shape=(1,), dtype="int32")

    w1_target_hot = Lambda(
        K.one_hot,
        arguments={"num_classes": vec_len},
        output_shape=(1, vec_len),
        name="w1_target"
    )(w1_target)

    w2_target_hot = Lambda(
        K.one_hot,
        arguments={"num_classes": vec_len},
        output_shape=(1, vec_len),
        name="w2_target"
    )(w2_target)

    w3_target_hot = Lambda(
        K.one_hot,
        arguments={"num_classes": vec_len},
        output_shape=(1, vec_len),
        name="w3_target"
    )(w3_target)

    w_input_hot = Lambda(
        K.one_hot,
        arguments={"num_classes": vec_len},
        output_shape=(1, vec_len),
        name="w1_one_hot"
    )(w_input)

    target_output = Maximum()([w1_target_hot, w2_target_hot, w3_target_hot])

    encoding_layer = Dense(128, activation="linear")(w_input_hot)

    prediction = Dense(vec_len, activation="tanh")(encoding_layer)

    model = Model(inputs=[w_input, w1_target, w2_target, w3_target], outputs=prediction)

    model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["categorical_accuracy"],
                  target_tensors=[target_output])

    h = model.fit(x=[trainings_set["target"], trainings_set["w1"], trainings_set["w2"], trainings_set["w3"]], verbose=1,
                  epochs=100, batch_size=1000)
