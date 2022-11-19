from initDatabase import *
from createModel import *
from predictor import *

X_test, X_train, y_test, y_train = init_database(1000)

newsVzer, newsTfidf, clf = create_model(X_test, X_train, y_test, y_train)

# True news
text = 'Qatar will host the world cup 2022'

pred = predictor(newsVzer, newsTfidf, clf, text)

print(pred)






