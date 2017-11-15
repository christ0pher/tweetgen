from keras import Input
from keras.engine import Model
from keras.layers import Lambda, K, Dense


def word2vec_embedding(vocab_length, vec_size):
    input_word = Input(shape=(1,), dtype="int32")

    input_word_hot = Lambda(
        K.one_hot,
        arguments={"num_classes": vocab_length},
        output_shape=(1, vocab_length),
        name="word_input_hot"
    )(input_word)

    encoding = Dense(vec_size, activation="linear", name="encoding")(input_word_hot)

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
