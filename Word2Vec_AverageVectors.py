import pandas as pd
import os
from nltk.corpus import stopwords
import nltk.data
import logging
import numpy as np  # Make sure that numpy is imported
from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestClassifier
from KaggleWord2VecUtility import KaggleWord2VecUtility

def makeFeatureVec(words, model, num_features):

    featureVec = np.zeros((num_features,), dtype="float32")

    nwords = 0.

    index2word_set = set(model.wv.index2word)

    for word in words:
        if word in index2word_set:
            nwords = nwords + 1.
            featureVec = np.add(featureVec, model[word])

    featureVec = np.divide(featureVec, nwords)
    return featureVec


def getAvgFeatureVecs(reviews, model, num_features):

    counter = 0.

    reviewFeatureVecs = np.zeros((len(reviews), num_features), dtype="float32")

    for review in reviews:

        if counter % 1000. == 0.:
            print
            "Review %d of %d" % (counter, len(reviews))

        reviewFeatureVecs[int(counter)] = makeFeatureVec(review, model, \
                                                         num_features)

        counter = counter + 1.
    return reviewFeatureVecs


def getCleanReviews(reviews):
    clean_reviews = []
    for review in reviews["review"]:
        clean_reviews.append(KaggleWord2VecUtility.review_to_wordlist(review, remove_stopwords=True))
    return clean_reviews

if __name__ == '__main__':

    train = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'labeledTrainData.tsv'), header=0,
                        delimiter="\t", quoting=3)
    test = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'testData.tsv'), header=0, delimiter="\t",
                       quoting=3)
    unlabeled_train = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', "unlabeledTrainData.tsv"), header=0,
                                  delimiter="\t", quoting=3)

    print
    "Read %d labeled train reviews, %d labeled test reviews, " \
    "and %d unlabeled reviews\n" % (train["review"].size,
                                    test["review"].size, unlabeled_train["review"].size)

    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    sentences = []  # Initialize an empty list of sentences

    print
    "Parsing sentences from training set"
    for review in train["review"]:
        sentences += KaggleWord2VecUtility.review_to_sentences(review, tokenizer)

    print
    "Parsing sentences from unlabeled set"
    for review in unlabeled_train["review"]:
        sentences += KaggleWord2VecUtility.review_to_sentences(review, tokenizer)

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', \
                        level=logging.INFO)

    num_features = 300  # Word vector dimensionality
    min_word_count = 40  # Minimum word count
    num_workers = 4  # Number of threads to run in parallel
    context = 10  # Context window size
    downsampling = 1e-3

    print
    "Training Word2Vec model..."
    model = Word2Vec(sentences, workers=num_workers, \
                     size=num_features, min_count=min_word_count, \
                     window=context, sample=downsampling, seed=1)

    model.init_sims(replace=True)

    model_name = "300features_40minwords_10context"
    model.save(model_name)

    model.doesnt_match("man woman child kitchen".split())
    model.doesnt_match("france england germany berlin".split())
    model.doesnt_match("paris berlin london austria".split())
    model.most_similar("man")
    model.most_similar("queen")
    model.most_similar("awful")

    print
    "Creating average feature vecs for training reviews"
    trainDataVecs = getAvgFeatureVecs(getCleanReviews(train), model, num_features)
    print
    "Creating average feature vecs for test reviews"
    testDataVecs = getAvgFeatureVecs(getCleanReviews(test), model, num_features)

    forest = RandomForestClassifier(n_estimators=100)

    print
    "Fitting a random forest to labeled training data..."
    forest = forest.fit(trainDataVecs, train["sentiment"])

    result = forest.predict(testDataVecs)

    output = pd.DataFrame(data={"id": test["id"], "sentiment": result})
    output.to_csv("Word2Vec_AverageVectors.csv", index=False, quoting=3)
    print
    "Wrote Word2Vec_AverageVectors.csv"
