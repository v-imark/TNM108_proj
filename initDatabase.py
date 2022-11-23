import string

import nltk
import pandas as pd
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split


def init_database(n):
    true_data = pd.read_csv('news/true/True.csv')
    fake_data = pd.read_csv('news/fake/Fake.csv')

    true_data = true_data.drop(['subject', 'date'], axis=1)
    fake_data = fake_data.drop(['subject', 'date'], axis=1)

    true_data = true_data.assign(target=1)
    fake_data = fake_data.assign(target=0)

    data = pd.merge(true_data[0:n], fake_data[0:n], how="outer")

    print('Data length', len(data))
    #data["title"] = data["title"].apply(app_stopwords)

    X_train, X_test, y_train, y_test = train_test_split(data["title"], data["target"], random_state=12)

    return X_train, X_test, y_train, y_test


def tfidf_vectorize_train(X, vzer, tfidfer):
    counts = vzer.fit_transform(X)
    tfidf = tfidfer.fit_transform(counts)

    return tfidf


def tfidf_vectorize_test(X, vzer: CountVectorizer, tfidfer):
    counts = vzer.transform(X)
    tfidf = tfidfer.transform(counts)

    return tfidf


def app_stopwords(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r"\\W", " ", text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text
