import sklearn.metrics
from sklearn.datasets import load_files
import pandas as pd

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
import nltk
from sklearn.naive_bayes import MultinomialNB


def init_database():
    true_data = pd.read_csv('news/true/True.csv')
    fake_data = pd.read_csv('news/fake/Fake.csv')

    #news = load_files(r'./news', shuffle=True)

    true_data = true_data.drop(['subject', 'date'], axis=1)
    fake_data = fake_data.drop(['subject', 'date'], axis=1)

    #news.data = news.data.drop(['subject', 'date'], axis=1)

    true_data = true_data.assign(target=1)
    fake_data = fake_data.assign(target=0)

    data = pd.merge(true_data, fake_data, how="outer")

    print(data.head())
    print(len(data))
    print(len(true_data))
    print(len(fake_data))

    STOPWORDS = stopwords.words('english')

    X_train, X_test, y_train, y_test = train_test_split(data["text"], data["target"], random_state=12)

    newsVzer = CountVectorizer(min_df=2, tokenizer=nltk.word_tokenize, max_features=3000)

    news_train_counts = newsVzer.fit_transform(X_train)

    newsTfidf = TfidfTransformer()

    news_train_tfdif = newsTfidf.fit_transform(news_train_counts)

    news_test_counts = newsVzer.transform(X_test)
    news_test_tfdif = newsTfidf.transform(news_test_counts)

    clf = MultinomialNB()
    clf.fit(news_train_tfdif, y_train)

    y_pred = clf.predict(news_test_tfdif)
    print(sklearn.metrics.accuracy_score(y_test,y_pred))

    news_new_counts = newsVzer.transform(['Russia won world war 2'])  # turn text into count vector
    news_new_tfidf = newsTfidf.transform(news_new_counts)  # turn into tfidf vector
    pred = clf.predict(news_new_tfidf)
    print(pred)

    return true_data


true = init_database()
