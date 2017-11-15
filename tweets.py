import re
import os
import ssl
import sys
import trainingsset
import vocabulary
import filepath
import time
from twitter import Twitter, OAuth
from twitter_credentials import CONFIG

if not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
    ssl._create_default_https_context = ssl._create_unverified_context


class CharStripper(dict):
    def __getitem__(self, k):
        el = chr(k)
        if el in "!$%&'()*+,-./:;<=>?[\]^_`{|}~\"":
            return None
        else:
            return k


def clean_tweet(tweet):
    tweet = re.sub("https://t.co/[a-zA-Z0-9]{10,10}", "", tweet)
    tweet = tweet.translate(CharStripper()).replace("\n", "")
    return tweet


def scrape_tweets(auth, user, num_tweets, output_file, batch_size=None, clean_func=None, encoding=None, callback=None):
    batch_size = 200 if batch_size is None else batch_size
    clean_func = clean_tweet if clean_func is None else clean_func
    encoding = "utf8" if encoding is None else encoding

    stream = Twitter(auth=auth, secure=True)
    with open(output_file, "w+", encoding=encoding) as rf:
        scraped_tweets = 0
        max_id = None
        while scraped_tweets < num_tweets:
            if max_id is None:
                results = stream.statuses.user_timeline(screen_name=user, count=batch_size)
            else:
                results = stream.statuses.user_timeline(screen_name=user, count=batch_size, max_id=max_id)

            if len(results) > 0:
                max_id = results[-1]["id"]

                for msg in results:
                    if "retweeted_status" not in msg:
                        tweet = clean_func(msg["text"])
                        rf.write(tweet + "\n")

                scraped_tweets += len(results)

                if callback is not None:
                    callback(scraped_tweets, num_tweets)
            else:
                break

            if results.rate_limit_remaining < 2:
                print("Scrape Limit Timeout %d seconds" % results.rate_limit_reset)
                time.sleep(results.rate_limit_reset)


if __name__ == "__main__":

    user = sys.argv[1]
    num_tweets = int(sys.argv[2])

    twitter_auth = OAuth(CONFIG["access_key"],
                         CONFIG["access_secret"],
                         CONFIG["consumer_key"],
                         CONFIG["consumer_secret"])

    print("Scraping Tweets of User %s ..." % user)
    scrape_tweets(twitter_auth, user, num_tweets, filepath.for_raw_text_file(user),
                  callback=lambda b, t: print("%d / %d" % (b, t)))
    print("DONE")

    print("Creating vocabulary list ...")
    vocab = vocabulary.create(filepath.for_raw_text_file(user))
    vocabulary.save(vocab, filepath.for_vocabulary_file(user))
    print("DONE")

    print("Creating n-gram trainings set ...")
    trainings_set = trainingsset.generate_ngram_to_next_word(filepath.for_raw_text_file(user), vocab)
    trainings_set.to_csv(filepath.for_trainings_set_file(user), index=False)
    print("DONE")
