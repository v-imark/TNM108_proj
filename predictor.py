def predictor(newsVzer, newsTfidf, clf, text):
    news_new_counts = newsVzer.transform([text])  # turn text into count vector
    news_new_tfidf = newsTfidf.transform(news_new_counts)  # turn into tfidf vector

    pred = clf.predict(news_new_tfidf)

    return pred
