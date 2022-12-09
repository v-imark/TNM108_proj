import tkinter as tk

from sklearn.feature_extraction.text import CountVectorizer

from predictor import *
from initDatabase import app_stopwords


class Gui:
    def __init__(self, master, vzer: CountVectorizer, tfidf, MNB, LR):
        self.master = master
        self.MNB_prediction = tk.StringVar()
        self.LR_prediction = tk.StringVar()
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

        self.MNB_answer_text = tk.Label(
            master=self.frame_body1,
            textvariable=self.MNB_prediction,
            font='Arial',
        )
        self.MNB_answer_text.pack()

        self.LR_answer_text = tk.Label(
            master=self.frame_body1,
            textvariable=self.LR_prediction,
            font='Arial',
        )
        self.LR_answer_text.pack()

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
            command=lambda: self.buttonpress(vzer, tfidf, MNB, LR)
        )
        self.button.pack()

        self.buttonClear = tk.Button(
            master=self.frame_body2,
            text='Clear text',
            width=15,
            height=3,
            bg='#f00',
            fg='#fff',
            border= 2,
            command=lambda: self.entry_text.delete("1.0", "end")
        )
        self.buttonClear.pack(pady=10)

        self.frame_title.pack()
        self.frame_body1.pack(pady=10)
        self.frame_body2.pack(pady=10)

    def buttonpress(self, vzer: CountVectorizer, tfidf, MNB, LR):
        text = self.entry_text.get("1.0", 'end-1c')
        # text = app_stopwords(text)
        prediction = predictor(vzer, tfidf, MNB, text)
        if prediction[0] == 0:
            print('Fake news')
            self.MNB_prediction.set('MNB: This is FAKE news')
            self.MNB_answer_text.config(fg='red')
        else:
            self.MNB_prediction.set('MNB: This is REAL news')
            self.MNB_answer_text.config(fg='green')

        prediction = predictor(vzer, tfidf, LR, text)
        if prediction[0] == 0:
            self.LR_prediction.set('RF: This is FAKE news')
            self.LR_answer_text.config(fg='red')
        else:
            self.LR_prediction.set('RF: This is REAL news')
            self.LR_answer_text.config(fg='green')

