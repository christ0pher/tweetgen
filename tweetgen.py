import vocabulary
import pandas as pd
import model
import filepath
from keras import callbacks

USER = "macbosse"


if __name__ == "__main__":
    trainings_set = pd.read_csv(filepath.for_trainings_set_file(USER))
    trainings_set = trainings_set.sample(frac=1).reset_index(drop=True)  # shuffle trainings data

    vocab = vocabulary.load(filepath.for_vocabulary_file(USER))
    vocab_len = len(vocab)

    w2v_weights = model.word2vec_load_encoding_weights(filepath.for_w2v_model(USER), vocab_len, 100)
    tweet_model = model.three_to_next_model(vocab_len, 100)

    tweet_model.get_layer("w1_encoding").set_weights(w2v_weights)
    tweet_model.get_layer("w2_encoding").set_weights(w2v_weights)
    tweet_model.get_layer("w3_encoding").set_weights(w2v_weights)

    print(tweet_model.summary())

    h = tweet_model.fit(
        x=[trainings_set["w1"], trainings_set["w2"], trainings_set["w3"], trainings_set["target"]],
        verbose=1,
        epochs=1000,
        batch_size=500,
        callbacks=[
            callbacks.TensorBoard(
                log_dir='./Graph',
                histogram_freq=0,
                write_graph=True,
                write_images=True
            ),
            callbacks.ModelCheckpoint(
                filepath.for_tweet_model(USER),
                monitor='loss',
                verbose=0,
                save_best_only=True,
                save_weights_only=False,
                mode='auto',
                period=1
            ),
            callbacks.EarlyStopping(monitor='loss', patience=5, verbose=0, mode='auto', min_delta=0.01)
        ]
    )

    tweet_model.save(filepath.for_tweet_model(USER))

    print(h)
