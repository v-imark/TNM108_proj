import pickle

from initDatabase import *
from createModel import *
from gui import *
import tkinter


# load the model from disk
MNB = pickle.load(open('multinomialNB.sav', 'rb'))
LR = pickle.load(open('linearRegression.sav', 'rb'))
newsVzer = pickle.load(open('vzer.sav', 'rb'))
newsTfidf = pickle.load(open('tfidf.sav', 'rb'))

root = tk.Tk()
root.geometry("800x750")
gui = Gui(root, newsVzer, newsTfidf, MNB, LR)
root.mainloop()







