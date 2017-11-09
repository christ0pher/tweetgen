import pandas as pd
from keras import callbacks
from keras_models.models import get_tweet_model, get_w2v_model

USER = "macbosse"
TRAIN_CSV = "./train_data/"+USER+".csv"
TRAIN_META_CSV = "./train_data/"+USER+"_meta.csv"
TRAIN_VOCAB_CSV = "./train_data/"+USER+"_vocab.csv"
MODEL_FILE = "./models/"+USER+".model"
WORD2VEC_TRAIN_MODEL = "./models/"+USER+"_w2v.model"


if __name__ == "__main__":

    meta_data = pd.read_csv(
        TRAIN_META_CSV
    )

    trainings_set = pd.read_csv(TRAIN_CSV)
    trainings_set = trainings_set.sample(frac=1).reset_index(drop=True)  # shuffle trainings data

    vocab_list = pd.read_csv(TRAIN_VOCAB_CSV)

    vec_len = meta_data["word_vec_len"][0]

    print(vec_len)

    w2v_model = get_w2v_model(vec_len)
    w2v_model.load_weights(WORD2VEC_TRAIN_MODEL, by_name=True)

    w2v_weights = w2v_model.get_layer("encoding_layer").get_weights()

    model = get_tweet_model(vec_len)

    model.get_layer("w1_w2v").set_weights(w2v_weights)
    model.get_layer("w2_w2v").set_weights(w2v_weights)
    model.get_layer("w3_w2v").set_weights(w2v_weights)

    print(model.summary())

    h = model.fit(x=[trainings_set["w1"], trainings_set["w2"], trainings_set["w3"], trainings_set["target"]],
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
                              MODEL_FILE,
                              monitor='loss',
                              verbose=0,
                              save_best_only=True,
                              save_weights_only=False,
                              mode='auto',
                              period=1
                          ),
                          callbacks.EarlyStopping(monitor='loss', patience=5, verbose=0, mode='auto', min_delta=0.01)
                      ])

    model.save(MODEL_FILE)

    print(h)
