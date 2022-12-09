import sklearn.metrics

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import nltk
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix



# Multinomial Naive Bayes
def mnb(y_test, y_train, train_tfidf, test_tfidf):
    clf = MultinomialNB()
    clf.fit(train_tfidf, y_train)

    y_pred = clf.predict(test_tfidf)
    print('Multinomial Naive Bayes:')
    print('Accuracy: ', sklearn.metrics.accuracy_score(y_test, y_pred))
    print('Confusion matrix: ', confusion_matrix(y_test, y_pred))
    return clf

#Random Forest Classfier
def lr(y_test, y_train, train_tfidf, test_tfidf):
    clf = RandomForestClassifier()
    clf.fit(train_tfidf, y_train)

    y_pred = clf.predict(test_tfidf)
    print('Random Forest:')
    print('Accuracy: ', sklearn.metrics.accuracy_score(y_test, y_pred))
    print('Confusion matrix: ', confusion_matrix(y_test, y_pred))
    return clf
