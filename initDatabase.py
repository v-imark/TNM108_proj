
import pandas as pd

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split


def init_database(n):
    true_data = pd.read_csv('news/true/True.csv')
    fake_data = pd.read_csv('news/fake/Fake.csv')

    true_data = true_data.drop(['subject', 'date'], axis=1)
    fake_data = fake_data.drop(['subject', 'date'], axis=1)

    true_data = true_data.assign(target=1)
    fake_data = fake_data.assign(target=0)

    data = pd.merge(true_data[0:n], fake_data[0:n], how="outer")

    print(data.head())
    print(len(data))

    STOPWORDS = stopwords.words('english')

    X_train, X_test, y_train, y_test = train_test_split(data["text"], data["target"], random_state=12)

    return X_train, X_test, y_train, y_test

