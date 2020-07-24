# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 14:47:25 2017
To extract, compute features to be used as input in training/applying models
@author: anthonyta
"""
import numpy as np
import nltk.data
import time
import logging
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from text_processing import text_to_sentences


def make_feature_doc2vec(description, model, num_features):
    # Function to average all of the word vectors in a given
    # paragraph
    #
    # Pre-initialize an empty numpy array(for speed)
    feature_vec = np.zeros((num_features,), dtype='float32')

    nsentences = 0

    # convert description to sentences
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    (sentences, tagged_words) = text_to_sentences(description,
                                                  tokenizer,
                                                  use_POSLem=True)
    # Loop over each setences in the description
    # and add its infered feature vector to the total
    for sentence in sentences:
        # get vector from model
        nsentences = nsentences + 1.
        feature_vec = np.add(feature_vec,
                             model.infer_vector(sentence))

    # Divide the result by the number of sentences to get the average
    if nsentences != 0:
        feature_vec = np.divide(feature_vec, sentences)
    return feature_vec


def get_avgfeature_vec_doc2vec(descriptions, model, num_features):
    # Given a set of descriptions(each one a list of sentences), calculate
    # the average feature vector for each one and return a 2D numpy array

    # Initialize a counter
    counter = 0

    # Preallocate a 2D numpy array, for speed
    descriptionfeature_vecs = np.zeros((len(descriptions), num_features),
                                       dtype='float32')
    # Loop through the descriptions
    for description in descriptions:
        if counter % 1000 == 0:
            logging.info('description %d of %d' % (counter, len(descriptions)))
        # Call the function(defined above) that makes average feature vectors
        descriptionfeature_vecs[counter] = makefeature_vec(description,
                                                           model,
                                                           num_features)

        # Increment the counter
        counter = counter + 1
    return descriptionfeature_vecs


def makefeature_vec(words, model, num_features):
    # Function to average all of the word vectors in a given
    # paragraph
    # Pre-initialize an empty numpy array(for speed)
    feature_vec = np.zeros((num_features,), dtype='float32')

    nwords = 0

    # Index2word is a list that contains the names of the words in
    # the model's vocabulary. Convert it to a set, for speed
    index2word_set = set(model.index2word)

    # Loop over each word in the description and, if it is in the model's
    # vocaublary, add its feature vector to the total
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1.
            feature_vec = np.add(feature_vec, model[word])

    # Divide the result by the number of words to get the average
    if nwords != 0:
        feature_vec = np.divide(feature_vec, nwords)
    return feature_vec


def get_avgfeature_vec(descriptions, model, num_features):
    # Given a set of descriptions(each one a list of words), calculate
    # the average feature vector for each one and return a 2D numpy array

    # Initialize a counter
    counter = 0

    # Preallocate a 2D numpy array, for speed
    descriptionfeature_vecs = np.zeros((len(descriptions), num_features),
                                       dtype='float32')

    # Loop through the descriptions
    for description in descriptions:
        if counter % 1000 == 0:
            logging.info('description %d of %d' % (counter, len(descriptions)))
        # Call the function(defined above) that makes average feature vectors
        descriptionfeature_vecs[counter] = makefeature_vec(description,
                                                           model,
                                                           num_features)
        # Increment the counter
        counter = counter + 1
    return descriptionfeature_vecs


def makefeature_vec_tfidf(words, model, vectorizer, vocab, num_features):
    # Function to average all of the word vectors in a given
    # paragraph using weights computed from tfidf
    # Pre-initialize an empty numpy array(for speed)
    feature_vec = np.zeros((num_features,), dtype='float32')
    # Index2word is a list that contains the names of the words in
    # the model's vocabulary. Convert it to a set, for speed
    index2word_set = set(model.index2word)
    # Loop over each word in the description and, if it is in the model's
    # vocaublary, add its feature vector to the total
    sum_weights = 0

    words_tfidf = vectorizer.transform([words]).toarray().reshape(len(vocab),)

    for word in words.split():
        if word in index2word_set:
            if word in vocab:
                weight = words_tfidf[vocab.index(word)]
                feature_vec = np.add(feature_vec, model[word] * weight)
                sum_weights += weight
    if sum_weights != 0:
        feature_vec = np.divide(feature_vec, sum_weights)

    return feature_vec


def get_avgfeature_vec_tfidf(descriptions, model, vectorizer, num_features):
    # Given a set of descriptions(each one a list of words), calculate
    # the average feature vector for each one using tfidf as weights
    # and return a 2D numpy array
    vocab = vectorizer.get_feature_names()

    counter = 0

    # Preallocate a 2D numpy array, for speed
    descriptionfeature_vecs = np.zeros((len(descriptions), num_features),
                                       dtype='float32')
    # Loop through the descriptions
    for description in descriptions:
        if counter % 1000 == 0:
            logging.info('description %d of %d' % (counter, len(descriptions)))
        # Call the function(defined above) that makes average feature vectors
        descriptionfeature_vecs[counter] = makefeature_vec_tfidf(description,
                                                                 model,
                                                                 vectorizer,
                                                                 vocab,
                                                                 num_features)

        # Increment the counter
        counter = counter + 1
    return descriptionfeature_vecs


def create_bag_of_centroids(wordlist, word_centroid_map):
    # The number of clusters is equal to the highest cluster index
    # in the word / centroid map
    num_centroids = max(word_centroid_map.values()) + 1
    # Pre-allocate the bag of centroids vector(for speed)
    bag_of_centroids = np.zeros(num_centroids, dtype='float32')
    # Loop over the words in the description. If the word is in the vocabulary,
    # find which cluster it belongs to, and increment that cluster count
    # by one
    for word in wordlist:
        if word in word_centroid_map:
            index = word_centroid_map[word]
            bag_of_centroids[index] += 1

    # Return the "bag of centroids"
    return bag_of_centroids


def create_centroid_vectors(model, descriptions):
    # Given a set of descriptions(each one a list of words), calculate
    # the centroid feature vector for each one
    # and return a 2D numpy array
    start = time.time()

    # Set "k"(num_clusters) to be 1/5th of the vocabulary size, or an
    # average of 5 words per cluster
    word_vectors = model.syn0
    num_clusters = word_vectors.shape[0] / 5

    # Initalize a k-means object and use it to extract centroids
    kmeans_clustering = KMeans(n_clusters=num_clusters)
    idx = kmeans_clustering.fit_predict(word_vectors)

    # Get the end time and print how long the process took
    end = time.time()
    elapsed = end - start
    logging.info('Time taken for K Means clustering: ', elapsed, 'seconds.')

    # Create a Word / Index dictionary, mapping each vocabulary word to
    # a cluster number
    word_centroid_map = dict(zip(model.index2word, idx))

    # Pre-allocate an array for the training set bags of centroids(for speed)
    train_centroids = np.zeros((len(descriptions), num_clusters),
                               dtype='float32')

    # Transform the training set descriptions into bags of centroids
    counter = 0
    for description in descriptions:
        train_centroids[counter] = create_bag_of_centroids(description,
                                                           word_centroid_map)
        counter += 1
    return train_centroids


def create_bow_vectors(descriptions):
    # Compute BOW vectors
    logging.info('Creating the bag of words...\n')
    # Initialize the "CountVectorizer" object, which is scikit-learn's
    # bag of words tool.
    vectorizer = CountVectorizer(analyzer='word',
                                 tokenizer=None,
                                 preprocessor=None,
                                 stop_words=None,
                                 max_features=500)

    train_data_features = vectorizer.fit_transform(descriptions)

    # Convert the result to an array
    train_data_features = train_data_features.toarray()
    return(vectorizer, train_data_features)


def create_tfidf_vectors(descriptions):
    # Compute Tfidf vectors
    logging.info('Creating the bag of words...\n')

    # Initialize the "TfidfVectorizer" object
    vectorizer = TfidfVectorizer(min_df=3,
                                 max_features=500,
                                 strip_accents='unicode',
                                 analyzer='word',
                                 token_pattern=r'\w{1,}',
                                 ngram_range=(1, 2),
                                 use_idf=1,
                                 smooth_idf=1,
                                 sublinear_tf=1,
                                 stop_words=None)

    train_data_features = vectorizer.fit_transform(descriptions)

    # Convert the result to an array
    train_data_features = train_data_features.toarray()
    return(vectorizer, train_data_features)
