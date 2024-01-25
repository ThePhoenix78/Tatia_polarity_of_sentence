import numpy as np
import pandas as pd

from to_table import make_table, invert_table

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from unicodedata import normalize

from textblob import TextBlob

import warnings

warnings.filterwarnings("ignore")


def filtre_message(message, code: str = "ascii"):
    message = message.replace("Ã©", "e").replace("Ãª", "e").replace("\t", "").replace("\n", "").replace("Ã‰", "e").replace("Ã´", "o")
    return normalize('NFD', message).encode(code, 'ignore').decode("utf8").strip()


colors = {
    "rouge": '\33[31m',
    "vert": '\33[32m',
    "bleu": '\33[34m',
    "blanc": '\33[37m'
}


labels = ["negative", "neutral", "positive"]


def color(text: str):
    text = str(text)
    if text in ["0", "1", "2"]:
        text = revert(int(text))

    c = "bleu"
    if text == "negative":
        c = "rouge"
    elif text == "positive":
        c = "vert"
    color = colors.get(c)
    blanc = colors.get("blanc")
    return f"{color}{text}{blanc}"


def convert(label):
    return labels.index(label)

def revert(label):
    return labels[label]


def calc_neutrality_word(text):
    dico = {
        "negative": 0,
        "neutral": 0,
        "positive": 0
        }

    balance = 0
    pol = 0

    for word in text.split():
        word = TextBlob(filtre_message(word))

        balance += word.sentiment.polarity

        if word.sentiment.polarity >= 0.5:
            dico["positive"] += 1

        elif word.sentiment.polarity <= -0.5:
            dico["negative"] += 1

        else:
            dico["neutral"] += .01

    key = max(dico, key=dico.get)
    if key == "negative":
        pol = 0
    elif key == "positive":
        pol = 2
    else:
        pol = 1

    if balance == 0:
        return pol , 1
    elif balance > 0:
        return pol, 2
    elif balance < 0:
        return pol, 0


def calc_neutrality_sentence(text):
    text = TextBlob(filtre_message(text))

    pol = 0

    for t in text.sentences:
        pol += t.sentiment.polarity

    if pol == 0:
        return 1
    elif pol > 0:
        return 2

    return 0

data = pd.read_csv("data.csv")


data['encoded'] = data['Sentiment'].apply(lambda x: labels.index(x))
data = data.drop("Sentiment", axis=1)

X, y = data["Sentence"], data["encoded"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

clf = LinearSVC()
clf.fit(X_train_vectorized, y_train)


a = clf.score(X_test_vectorized, y_test)
print("Accuracy : ", a)


def do_prediction(text):
    vectorized_text = vectorizer.transform([text])
    return clf.predict(vectorized_text)[0]


for i in range(10):
    text = X_test.iloc[i]
    p = do_prediction(text)
    q, s = calc_neutrality_word(text)
    r = calc_neutrality_sentence(text)

    print(f"{text} | REALITY {color(y_test.iloc[i])} | PONDERATE (word) {color(q)} ({color(s)}) | PONDERATE (sentence) {color(r)} | PREDICTION {color(p)}\n")

i = 0
word = []
word_p = []
sentence = []
ai = []

test = []

for text in X_test.iloc:
    # text = X_test.iloc[i]
    p = do_prediction(text)
    ai.append(p)

    q, s = calc_neutrality_word(text)
    r = calc_neutrality_sentence(text)

    word.append(round(q, 1))
    word_p.append(s)
    sentence.append(r)
    test.append(y_test.iloc[i])

    # print(y_test.iloc[i], p, q, s, r)
    i += 1

results = [[revert(i) for i in test][:10], [revert(i) for i in word][:10], [revert(i) for i in word_p][:10], [revert(i) for i in sentence][:10], [revert(i) for i in ai][:10]]

results = invert_table(results)


print(make_table(rows=results, labels=["text", "word", "word (o)", "sentence", "ai"], centered=True))


def plot_confusion_matrix(y_test, y_pred, labels=None, path: str = 'confusion_matrix.png'):
    confusion = confusion_matrix(y_true=y_test, y_pred=y_pred, normalize='true')

    # Create confusion matrix display
    cm_display = ConfusionMatrixDisplay(confusion, display_labels=labels)

    # Plot and save the figure
    fig, ax = plt.subplots(figsize=(8, 8))

    cm_display.plot(ax=ax)
    plt.title('Confusion matrix (normalized)')
    fig.savefig(path)


"""
plot_confusion_matrix(word, test, path="word_prediction.png", labels=["negative", "neutral", "positive"])
plot_confusion_matrix(word_p, test, path="word2_prediction.png", labels=["negative", "neutral", "positive"])
plot_confusion_matrix(sentence, test, path="sentence_prediction.png", labels=["negative", "neutral", "positive"])
plot_confusion_matrix(ai, test, path="ai_prediction.png", labels=["negative", "neutral", "positive"])
"""
