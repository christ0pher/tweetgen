from sklearn.cluster import KMeans

__author__ = 'christopher@levire.com'
import numpy


def get_kmeans(vocab_list, w2v_model):
    global word, p_w, word_centers, h
    elems = []
    for word in vocab_list["word"]:
        p_w = w2v_model.predict(x=numpy.array([vocab_list[vocab_list["word"] == word]["index"].values[0]]))
        elems.append(p_w.flatten())
    dataset = numpy.array(elems)
    kmeans = KMeans(init="random", n_clusters=15)
    kmeans.fit(dataset)

    word_centers = {}
    look_up_class = {}
    for word in vocab_list["word"]:
        p_w = w2v_model.predict(x=numpy.array([vocab_list[vocab_list["word"] == word]["index"].values[0]]))
        klass = kmeans.predict(p_w.flatten().reshape(1, -1))[0]
        h = str(klass)
        if h not in word_centers:
            word_centers[h] = []
        word_centers[h].append(word)
        look_up_class[vocab_list[vocab_list["word"] == word]["index"].values[0]] = klass
    return word_centers, look_up_class