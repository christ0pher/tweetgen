import numpy
from sklearn.cluster import KMeans


def get_kmeans(vocab_list, w2v_model):
    vectors = []
    for word in vocab_list["word"]:
        vec = w2v_model.predict(x=numpy.array([vocab_list[vocab_list["word"] == word]["index"].values[0]]))
        vectors.append(vec.flatten())

    kmeans = KMeans(init="random", n_clusters=15)
    kmeans.fit(numpy.array(vectors))

    word_centers = {}
    clusters = {}
    for word in vocab_list["word"]:
        vec = w2v_model.predict(x=numpy.array([vocab_list[vocab_list["word"] == word]["index"].values[0]]))
        cluster = kmeans.predict(vec.flatten().reshape(1, -1))[0]
        h = str(cluster)
        if h not in word_centers:
            word_centers[h] = []
        word_centers[h].append(word)
        clusters[vocab_list[vocab_list["word"] == word]["index"].values[0]] = cluster
    return word_centers, clusters
