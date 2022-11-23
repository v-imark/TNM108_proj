import sklearn.metrics

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import nltk
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB


# Multinomial Naive Bayes
def mnb(y_test, y_train, train_tfidf, test_tfidf):
    clf = MultinomialNB()
    clf.fit(train_tfidf, y_train)

    y_pred = clf.predict(test_tfidf)
    print('MNB accuracy:', sklearn.metrics.accuracy_score(y_test, y_pred))

    return clf


# Logistic Regression
def lr(y_test, y_train, train_tfidf, test_tfidf):
    clf = LogisticRegression()
    clf.fit(train_tfidf, y_train)

    y_pred = clf.predict(test_tfidf)
    print('LR accuracy:', sklearn.metrics.accuracy_score(y_test, y_pred))

    return clf
