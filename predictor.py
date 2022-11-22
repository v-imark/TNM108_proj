from initDatabase import *


def predictor(newsVzer, newsTfidf: CountVectorizer, clf, text):
    print(text)
    tfidf_text = tfidf_vectorize_test([text], newsVzer, newsTfidf)

    pred = clf.predict(tfidf_text)
    print(pred)
    return pred
