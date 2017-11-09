import re
import os
import ssl
from twitter import Twitter, OAuth
from twitter_credentials import CONFIG

BATCHES = 180
USER = "khloekardashian"
FRQ_CAP = 2
TRAIN_CSV = "./train_data/"+USER+".csv"
TRAIN_METADATA = "./train_data/"+USER+"_meta.csv"
TRAIN_VOCAB = "./train_data/"+USER+"_vocab.csv"
RAW_TEXT = "./raw_data/"+USER+".txt"


class CharStripper(dict):
    def __getitem__(self, k):
        el = chr(k)
        if el in "!$%&'()*+,-./:;<=>?[\]^_`{|}~\"":
            return None
        else:
            return k


if not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
    ssl._create_default_https_context = ssl._create_unverified_context

if __name__ == "__main__":

    twitter_auth = OAuth(CONFIG["access_key"],
                         CONFIG["access_secret"],
                         CONFIG["consumer_key"],
                         CONFIG["consumer_secret"])

    stream = Twitter(auth=twitter_auth, secure=True)
    results = stream.statuses.user_timeline(screen_name=USER, count=200)

    remover = "https://t.co/[a-zA-Z0-9]{10,10}"

    rate_limit = results.rate_limit_remaining
    with open(RAW_TEXT, "w+") as rf:
        for limit in range(1, int(BATCHES)):

            for msg in results:
                if "retweeted_status" not in msg:
                    txt = re.sub(remover, '', msg["text"])
                    txt = txt.translate(CharStripper()).replace("\n", "")
                    rf.write(txt+"\n")

            results = stream.statuses.user_timeline(screen_name=USER, count=200, max_id=results[-1]["id"])

    vocab_list = {}
    frequencies = {}

    with open(RAW_TEXT) as input_file:
        one_hot_index = 0
        for line in input_file:
            for word in line.split():
                w = word.lower()
                if w not in vocab_list:
                    vocab_list[w] = one_hot_index
                    frequencies[w] = 0
                frequencies[w] += 1

    for k, v in frequencies.items():
        if v <= FRQ_CAP:
            del vocab_list[k]

    rvocab_list = {v: k for k, v in vocab_list.items()}

    print(len(vocab_list))

    one_hot_index = 0
    for k in vocab_list.keys():
        vocab_list[k] = str(one_hot_index)
        one_hot_index += 1
    print(vocab_list)

    # Create trainings set
    m = 0
    with open(TRAIN_CSV, "w+") as output_file:
        output_file.write("w1,w2,w3,target\n")
        with open(RAW_TEXT) as input_file:

            for line in input_file:
                words = line.split()
                for i in range(0, len(words) - 3):
                    w1, w2, w3, target = words[i].lower(), words[i + 1].lower(), words[i + 2].lower(), words[
                        i + 3].lower()
                    if w1 in vocab_list and w2 in vocab_list and w3 in vocab_list and target in vocab_list:
                        output_file.write(
                            vocab_list[w1] + "," + vocab_list[w2] + "," + vocab_list[w3] + "," + vocab_list[
                                target] + "\n")
                        m += 1

    with open(TRAIN_VOCAB, "w+") as meta_file:
        meta_file.write("word,index\n")
        for k, v in vocab_list.items():
            meta_file.write(str(k) + "," + str(v)+"\n")

    with open(TRAIN_METADATA, "w+") as meta_file:
        meta_file.write("m, features, word_vec_len\n")
        meta_file.write(str(1)+","+str(1)+","+str(len(vocab_list))+",")
