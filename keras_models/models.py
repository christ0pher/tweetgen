from keras import Input
from keras.engine import Model
from keras.layers import Lambda, Dense, Concatenate, K, LSTM

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
    """
    word_embedding = Dense(1024, activation="linear")
    w1_embedding = word_embedding(w1_one_hot)
    w2_embedding = word_embedding(w2_one_hot)
    w3_embedding = word_embedding(w3_one_hot)
    
    embedding = Concatenate()([w1_embedding, w2_embedding, w3_embedding])
    """
    w1_w2v_trained = Dense(100, activation="linear", name="w1_w2v")
    w1_w2v_trained.trainable = False
    w1_w2v_trained = w1_w2v_trained(w1_one_hot)

    w2_w2v_trained = Dense(100, activation="linear", name="w2_w2v")
    w2_w2v_trained.trainable = False
    w2_w2v_trained = w2_w2v_trained(w2_one_hot)

    w3_w2v_trained = Dense(100, activation="linear", name="w3_w2v")
    w3_w2v_trained.trainable = False
    w3_w2v_trained = w3_w2v_trained(w3_one_hot)
    """
    mem_1 = LSTM(64)(w1_w2v_trained)
    mem_2 = LSTM(64)(w2_w2v_trained)
    mem_3 = LSTM(64)(w3_w2v_trained)

    memento_embedding = Concatenate()([mem_1, mem_2, mem_3])
    """
    w1_learn = Dense(256, activation="linear")(w1_w2v_trained)
    w2_learn = Dense(256, activation="linear")(w2_w2v_trained)
    w3_learn = Dense(256, activation="linear")(w3_w2v_trained)

    combined_learner = Dense(1500, activation="sigmoid")
    w1 = combined_learner(w1_learn)
    w2 = combined_learner(w2_learn)
    w3 = combined_learner(w3_learn)

    embedding = Concatenate()([w1, w2, w3])

    hidden = Dense(512, activation="sigmoid")(embedding)
    prediction_layer = Dense(vec_len, activation="softmax")(hidden)
    model = Model(inputs=[w1_input, w2_input, w3_input, target_input], outputs=prediction_layer)
    model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["categorical_accuracy"],
                  target_tensors=[target_one_hot])

    return model