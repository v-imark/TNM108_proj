import sklearn.metrics

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import nltk
from sklearn.naive_bayes import MultinomialNB


def create_model(X_test, X_train, y_test, y_train):
    newsVzer = CountVectorizer(min_df=2, tokenizer=nltk.word_tokenize, max_features=3000)
    news_train_counts = newsVzer.fit_transform(X_train)

    newsTfidf = TfidfTransformer()
    news_train_tfdif = newsTfidf.fit_transform(news_train_counts)

    news_test_counts = newsVzer.transform(X_test)
    news_test_tfdif = newsTfidf.transform(news_test_counts)

    clf = MultinomialNB()
    clf.fit(news_train_tfdif, y_train)

    y_pred = clf.predict(news_test_tfdif)
    print(sklearn.metrics.accuracy_score(y_test, y_pred))

    return newsVzer, newsTfidf, clf

