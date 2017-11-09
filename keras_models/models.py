from keras import Input
from keras.engine import Model
from keras.layers import Lambda, Dense, Concatenate, K

__author__ = 'christopher@levire.com'


def get_w2v_model(vec_len):
    global loaded_model
    word_input = Input(shape=(1,), dtype="int32")
    word_input_hot = Lambda(
        K.one_hot,
        arguments={"num_classes": vec_len},
        output_shape=(1, vec_len),
        name="w1_input_hot"
    )(word_input)
    encoding_layer_predict = Dense(100, activation="linear", name="encoding_layer")(word_input_hot)
    loaded_model = Model(inputs=[word_input], outputs=encoding_layer_predict)
    loaded_model.compile(optimizer="sgd", loss="categorical_crossentropy",
                         metrics=["categorical_accuracy", 'binary_accuracy'],
                         target_tensors=[encoding_layer_predict])

    return loaded_model


def get_tweet_model(vec_len):
    global model
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
    model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["categorical_accuracy"],
                  target_tensors=[target_one_hot])

    return model