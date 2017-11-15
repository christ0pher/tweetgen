def for_raw_text_file(user):
    return "./raw_data/" + user + ".txt"


def for_vocabulary_file(user):
    return for_trainings_set_file(user, "vocab")


def for_skip_gram_file(user):
    return for_trainings_set_file(user, "skipgram")


def for_trainings_set_file(user, suffix=None):
    if suffix is None:
        return "./train_data/" + user + ".csv"
    else:
        return "./train_data/" + user + "_" + suffix + ".csv"


def for_w2v_model(user):
    return "./models/"+user+"_w2v.model"


def for_tweet_model(user):
    return "./models/"+user+".model"
