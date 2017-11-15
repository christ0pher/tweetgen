from keras import Input
from keras.engine import Model
from keras.layers import Lambda, K, Dense, Concatenate


def word2vec_embedding(vocab_length, vec_size, name=None, trainable=None):
    trainable = True if trainable is None else trainable
    name = "word" if name is None else name

    input_word = Input(shape=(1,), dtype="int32")

    input_word_hot = Lambda(
        K.one_hot,
        arguments={"num_classes": vocab_length},
        output_shape=(1, vocab_length),
        name="%s_input_hot" % name
    )(input_word)

    encoding = Dense(vec_size, activation="linear", name="%s_encoding" % name)(input_word_hot)
    encoding.trainable = trainable

    return input_word, encoding


def word2vec(vocab_length, vec_size):
    input_word, encoding = word2vec_embedding(vocab_length, vec_size)
    model = Model(inputs=[input_word], outputs=encoding)
    model.compile(
        optimizer="sgd",
        loss="categorical_crossentropy",
        metrics=["categorical_accuracy", 'binary_accuracy']
    )

    return model


def word2vec_load_encoding_weights(filepath, vocab_length, vec_size, name=None):
    name = "word" if name is None else name
    w2v_model = word2vec(vocab_length, vec_size)
    w2v_model.load_weights(filepath, by_name=True)
    return w2v_model.get_layer("%s_encoding" % name).get_weights()


def three_to_next_model(vocab_length, vec_size):
    w1_input, w1_encoding = word2vec_embedding(vocab_length, vec_size, name="w1", trainable=False)
    w2_input, w2_encoding = word2vec_embedding(vocab_length, vec_size, name="w2", trainable=False)
    w3_input, w3_encoding = word2vec_embedding(vocab_length, vec_size, name="w3", trainable=False)

    target_input = Input(shape=(1,), dtype="int32")

    target_one_hot = Lambda(
        K.one_hot,
        arguments={"num_classes": vocab_length},
        output_shape=(1, vocab_length),
        name="target_one_hot"
    )(target_input)

    w1_learn = Dense(256, activation="softmax")(w1_encoding)
    w2_learn = Dense(256, activation="softmax")(w2_encoding)
    w3_learn = Dense(256, activation="softmax")(w3_encoding)

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

    prediction_layer = Dense(vocab_length, activation="softmax")(embedding)
    model = Model(inputs=[w1_input, w2_input, w3_input, target_input], outputs=prediction_layer)
    model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["categorical_accuracy"],
                  target_tensors=[target_one_hot])

    return model
