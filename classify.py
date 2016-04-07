#!/usr/bin/env python
# -*- coding: utf-8 -*-


import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')

import os, json, sys
import csv
import codecs
import time
import re
from util import full_stack, chunks, md5

import sklearn
import sklearn.ensemble
import sklearn.naive_bayes
import sklearn.metrics
import sklearn.utils

import numpy as np
#np.set_printoptions(threshold=np.nan)
import math

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import common

def determine_class(tweet, mode = 'find_irrelevant'):

    l = int(tweet['class']) if tweet['class'] else None#common.determine_class(tweet)

    if (l==None):
        return None

    if (mode == 'find_irrelevant'):
        return 1 if l == -1 else 0
    elif (mode == 'find_smoker'):
        if (l == 1):
            return 1
        elif (l == -1): # ignore irrelevant tweets
            return None
        else:
            return 0
    else:
        return None

def generate_training_feature_matrix(labeled_tweets, mode ='find_irrelevant'):

    # with open(labeled_transgender_identification_csv_filename, 'r', newline='', encoding='utf-8') as rf:
    #     reader = csv.DictReader(rf)

    training_tweets = []
    tweets_id = set()
    y = []
    has_url = []
    has_username = []
    for tweet in labeled_tweets:
        tweet_id = int(float(tweet['id']))

        c = determine_class(tweet, mode=mode)
        if (c == None):
            continue
        y.append(c)

        text = tweet['text']

        has_url.append(common.has_url(text))
        has_username.append(common.has_username(text))

        text = common.sanitize_text(text)

        training_tweets.append(text)
        tweets_id.add(tweet_id)

    logger.info('positive: %d; negative: %d'%(y.count(1), y.count(0)))

    # ngram_range=(1, 2),token_pattern=r'\b\w+\b', min_df=1, , max_features = 5000, binary = False
    # TfidfVectorizer(min_df=1)
    vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1, max_features = 5000)#CountVectorizer(analyzer = "word", ngram_range=(1, 2), binary = True, min_df=1, tokenizer = None, preprocessor = None, stop_words = None)
    X = vectorizer.fit_transform(training_tweets)
    X = X.toarray()

    X = np.column_stack((X, has_username, has_url))

    # vocab = vectorizer.get_feature_names()

    # # Sum up the counts of each vocabulary word
    # dist = np.sum(train_data_features, axis=0)

    # # For each, print the vocabulary word and the number of times it
    # # appears in the training set
    # for tag, count in zip(vocab, dist):
    #     print(count, tag)
    return X, np.array(y), vectorizer, tweets_id

def random_forest(X, y):
    best_accuracy = 0.0

    n_estimators = 100

    while(True):

        cfr = sklearn.ensemble.RandomForestClassifier(n_estimators=n_estimators, max_features='sqrt', max_depth=None, min_samples_split=1, n_jobs=-1)

        #Simple K-Fold cross validation. 10 folds.
        cv = sklearn.cross_validation.StratifiedKFold(y, n_folds=10)

        avg_accuracy = 0.0
        avg_precision = 0.0 #avg over cv, not average precision score (area under precision-recall curve)
        avg_recall = 0.0
        for train_index, test_index in cv:

            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            #logger.info(y_train)
            #probas = cfr.fit(X_train, y_train).predict_proba(X_test)
            model = cfr.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            y_score = model.predict_proba(X_test)

            avg_accuracy += sklearn.metrics.accuracy_score(y_test, y_pred)
            avg_precision += sklearn.metrics.precision_score(y_test, y_pred)
            avg_recall += sklearn.metrics.recall_score(y_test, y_pred)
            #logger.info(sklearn.metrics.roc_auc_score(y_test, y_score, average='weighted'))
            #logger.info(model.predict_proba(X_test))

        avg_accuracy /= len(cv)
        avg_precision /= len(cv)
        avg_recall /= len(cv)

        if (avg_accuracy - best_accuracy) < 0:
            logger.info('best accuracy: %.3f; precision: %.3f; recall: %.3f'%(avg_accuracy, avg_precision, avg_recall))
            break

        if (avg_accuracy > best_accuracy):
            best_accuracy = avg_accuracy

        n_estimators += 10

    logger.info("best n_estimators: [%d]"%(n_estimators))

    fr = sklearn.ensemble.RandomForestClassifier(n_estimators=n_estimators, max_features='sqrt', max_depth=None, min_samples_split=1, n_jobs=-1)

    return fr.fit(X, y)

def classify_tweets(labeled_tweets, testing_tweets, mode="find_irrelevant"):

    X, y, vectorizer,labeled_tweets_id = generate_training_feature_matrix(labeled_tweets, mode=mode)

    model = random_forest(X, y)

    resulting_tweets = []
    tweets = []
    tweets_txt = []

    has_url = []
    has_username = []


    for tweet in testing_tweets:
        tweet_id = int(float(tweet['id']))
        if (tweet_id in labeled_tweets_id):
            continue
        # already classified
        if (tweet['class']):
            resulting_tweets.append(tweet)
            continue

        text = tweet['text']

        has_url.append(common.has_url(text))
        has_username.append(common.has_username(text))

        text = common.sanitize_text(text)

        tweets_txt.append(text)
        tweets.append(tweet)

    logger.info('already labeled: [%d]; testing: [%d]'%(len(resulting_tweets), len(tweets)))

    X_test = vectorizer.transform(tweets_txt)
    X_test = X_test.toarray()
    
    X_test = np.column_stack((X_test, has_username, has_url))

    y_pred = model.predict(X_test)

    #logger.info('prediction: [%d]'%(len(y_pred)))
    for c, tweet in zip(y_pred, tweets):

        if (mode == 'find_smoker'):
            tweet['class'] = c
        elif (mode == 'find_irrelevant'):
            tweet['class'] = -1 if c == 1 else 0

        resulting_tweets.append(tweet)

    return resulting_tweets


def classify(csv_filename, labeled_csv_filenames = []):

    labeled_tweets_id = set()
    labeled_tweets = []
    for labeled_csv_filename in labeled_csv_filenames:
        with open(labeled_csv_filename, 'r', newline='', encoding='utf-8') as rf:
            reader = csv.DictReader(rf)

            for tweet in reader:
                tweet_id = int(float(tweet['id']))
                labeled_tweets_id.add(tweet_id)
                labeled_tweets.append(tweet)

    testing_tweets = []
    with open(csv_filename, 'r', newline='', encoding='utf-8') as rf:
        reader = csv.DictReader(rf)

        for tweet in reader:
            tweet_id = int(float(tweet['id']))
            if (tweet_id in labeled_tweets_id):
                continue
            testing_tweets.append(tweet)

    resulting_tweets = classify_tweets(labeled_tweets, testing_tweets, mode='find_irrelevant')


    for tweet in resulting_tweets:
        if (int(tweet['class']) == 0):
            tweet['class'] = ''

    resulting_tweets = classify_tweets(labeled_tweets, resulting_tweets, mode='find_smoker')

    tweet_cnt = {
        'smoker': 0,
        'non-smoker': 0
    }

    user_cnt = {
        'smoker': set(),
        'non-smoker': set()
    }

    for tweet in resulting_tweets:
        if (int(tweet['class']) == 1):
            tweet_cnt['smoker'] += 1
            user_cnt['smoker'].add(int(float(tweet['uid'])))

        if (int(tweet['class']) == 0):
            tweet_cnt['non-smoker'] += 1
            user_cnt['non-smoker'].add(int(float(tweet['uid'])))

    logger.info("TWEETS: smoker: %d: exclude-smoker: %d; total: %d"%(tweet_cnt['smoker'], tweet_cnt['non-smoker'], tweet_cnt['smoker'] + tweet_cnt['non-smoker']))
    logger.info("USERS: smoker: %d: exclude-smoker: %d; total: %d"%(len(user_cnt['smoker']), len(user_cnt['non-smoker']), len(user_cnt['smoker']) + len(user_cnt['non-smoker'])))


    output_fieldnames = ['class', 'comment', 'terms', 'text', 'id', 'created_at', 'location', 'state', 'uid', 'friends_count', 'followers_count', 'user_created_at', 'statuses_count']

    with open('predicted.%s'%csv_filename, 'w', newline='', encoding='utf-8') as wf:

        writer = csv.DictWriter(wf, fieldnames=output_fieldnames, delimiter=',', quoting=csv.QUOTE_ALL)
        writer.writeheader()

        for tweet in resulting_tweets:
            writer.writerow(tweet)

if __name__ == "__main__":

    logger.info(sys.version)

    classify("./data/tobacco_related_tweets.csv", ["./data/tobacco_related_tweets.annotated.csv"])
