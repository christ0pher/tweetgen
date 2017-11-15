from keras import Input
from keras.engine import Model
from keras.layers import Lambda, Dense, Concatenate, K


def get_tweet_model(vec_len):
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

    w1_w2v_trained = Dense(100, activation="linear", name="w1_w2v")
    w1_w2v_trained.trainable = False
    w1_w2v_trained = w1_w2v_trained(w1_one_hot)

    w2_w2v_trained = Dense(100, activation="linear", name="w2_w2v")
    w2_w2v_trained.trainable = False
    w2_w2v_trained = w2_w2v_trained(w2_one_hot)

    w3_w2v_trained = Dense(100, activation="linear", name="w3_w2v")
    w3_w2v_trained.trainable = False
    w3_w2v_trained = w3_w2v_trained(w3_one_hot)

    w1_learn = Dense(256, activation="softmax")(w1_w2v_trained)
    w2_learn = Dense(256, activation="softmax")(w2_w2v_trained)
    w3_learn = Dense(256, activation="softmax")(w3_w2v_trained)

    w1_w2_combined = Dense(256, activation="sigmoid")
    w1_w2_combined_1 = w1_w2_combined(w1_learn)
    w1_w2_combined_2 = w1_w2_combined(w2_learn)

    w2_w3_combined = Dense(256, activation="sigmoid")
    w2_w3_combined_1 = w2_w3_combined(w2_learn)
    w2_w3_combined_2 = w2_w3_combined(w3_learn)

    w1_w3_combined = Dense(256, activation="sigmoid")
    w1_w3_combined_1 = w1_w3_combined(w1_learn)
    w1_w3_combined_2 = w1_w3_combined(w3_learn)

    combined_learner = Dense(512, activation="sigmoid")
    w1_w2_1 = combined_learner(w1_w2_combined_1)
    w1_w2_2 = combined_learner(w1_w2_combined_2)
    w2_w3_1 = combined_learner(w2_w3_combined_1)
    w2_w3_2 = combined_learner(w2_w3_combined_2)
    w1_w3_1 = combined_learner(w1_w3_combined_1)
    w1_w3_2 = combined_learner(w1_w3_combined_2)

    embedding = Concatenate()([w1_w2_1, w1_w2_2, w2_w3_1, w2_w3_2, w1_w3_1, w1_w3_2])

    prediction_layer = Dense(vec_len, activation="softmax")(embedding)
    model = Model(inputs=[w1_input, w2_input, w3_input, target_input], outputs=prediction_layer)
    model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["categorical_accuracy"],
                  target_tensors=[target_one_hot])

    return model
