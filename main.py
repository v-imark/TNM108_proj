from initDatabase import *
from createModel import *
from gui import *
import tkinter

X_test, X_train, y_test, y_train = init_database(1000)

newsVzer = CountVectorizer(min_df=2, tokenizer=nltk.word_tokenize, stop_words='english', max_features=3000)
newsTfidf = TfidfTransformer()

train_tfidf = tfidf_vectorize_train(X_train, newsVzer, newsTfidf)
test_tfidf = tfidf_vectorize_test(X_test, newsVzer, newsTfidf)

MNB = mnb(y_test, y_train, train_tfidf, test_tfidf)
LR = lr(y_test, y_train, train_tfidf, test_tfidf)

root = tk.Tk()
gui = Gui(root, newsVzer, newsTfidf, LR)
root.mainloop()







