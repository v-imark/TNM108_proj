from initDatabase import *
from createModel import *
import pickle

X_test, X_train, y_test, y_train = init_database()

newsVzer = CountVectorizer(min_df=2, tokenizer=nltk.word_tokenize, max_features=3000)
newsTfidf = TfidfTransformer()

train_tfidf = tfidf_vectorize_train(X_train, newsVzer, newsTfidf)
test_tfidf = tfidf_vectorize_test(X_test, newsVzer, newsTfidf)

MNB = mnb(y_test, y_train, train_tfidf, test_tfidf)
LR = lr(y_test, y_train, train_tfidf, test_tfidf)


# save the model to disk
filename = 'multinomialNB.sav'
pickle.dump(MNB, open(filename, 'wb'))
filename = 'linearRegression.sav'
pickle.dump(LR, open(filename, 'wb'))
filename = 'vzer.sav'
pickle.dump(newsVzer, open(filename, 'wb'))
filename = 'tfidf.sav'
pickle.dump(newsTfidf, open(filename, 'wb'))
