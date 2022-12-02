import tkinter as tk

from sklearn.feature_extraction.text import CountVectorizer

from predictor import *
from initDatabase import app_stopwords


class Gui:
    def __init__(self, master, vzer: CountVectorizer, tfidf, clf, ):
        self.master = master
        self.prediction = tk.StringVar()
        self.frame_title = tk.Frame(master)
        self.frame_body1 = tk.Frame(master)
        self.frame_body2 = tk.Frame(master)

        self.master.title("Real or Fake News?")

        self.label_header = tk.Label(
            master=self.frame_title,
            text="Real or FAKE news?",
            font=("Arial", 25),
        )
        self.label_header.pack()

        self.answer_text = tk.Label(
            master=self.frame_body1,
            textvariable=self.prediction,
            font='Arial',
        )
        self.answer_text.pack()

        self.entry_text = tk.Text(
            master=self.frame_body2,
            height=25,
        )
        self.entry_text.pack(pady=10, padx=10)

        self.button = tk.Button(
            master=self.frame_body2,
            text='Predict',
            width=25,
            height=5,
            command=lambda: self.buttonpress(vzer, tfidf, clf)
        )
        self.button.pack()

        self.frame_title.pack()
        self.frame_body1.pack(pady=10)
        self.frame_body2.pack(pady=10)

    def buttonpress(self, vzer: CountVectorizer, tfidf, clf):
        text = self.entry_text.get("1.0", 'end-1c')
        # text = app_stopwords(text)
        prediction = predictor(vzer, tfidf, clf, text)
        if prediction[0] == 0:
            print('Fake news')
            self.prediction.set('This is FAKE news')
            self.answer_text.config(fg='red')
        else:
            print('Real News')
            self.prediction.set('This is REAL news')
            self.answer_text.config(fg='green')
