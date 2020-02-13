import math
import os
import random
from collections import Counter
import numpy as np
import pandas as pd
from flask import Flask
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error

classifiers = {
    "logistical_regression": {
        "classifier": LogisticRegression(max_iter=500),
        "vectorizer": None
    },
    "decision_tree_classifier": {
        "classifier": DecisionTreeClassifier(),
        "vectorizer": None
    }
}

app = Flask(__name__)


def entropy(s):
    p, lns = Counter(s), float(len(s))
    return -sum(count / lns * math.log(count / lns, 2) for count in p.values())


def getTokens(input):
    tokens_by_slash = str(input.encode('utf-8')).split('/')  # get tokens after splitting by slash
    all_tokens = []
    for i in tokens_by_slash:
        tokens = str(i).split('-')  # get tokens after splitting by dash
        tokens_by_dot = []
        for j in range(0, len(tokens)):
            temp_tokens = str(tokens[j]).split('.')  # get tokens after splitting by dot
            tokens_by_dot = tokens_by_dot + temp_tokens
        all_tokens = all_tokens + tokens + tokens_by_dot
    all_tokens = list(set(all_tokens))  # remove redundant tokens
    if 'com' in all_tokens:
        all_tokens.remove(
            'com')  # removing .com since it occurs a lot of times and it should not be included in our features
    return all_tokens


def train_classifier():
    allurls = '/home/maxime/PycharmProjects/iot_security/data/data.csv'  # path to our all urls file
    allurldata = np.array(pd.DataFrame(pd.read_csv(allurls, ',', error_bad_lines=False)))  # converting to a numpy array
    random.shuffle(allurldata)  # shuffling

    y = [int(d[1] == "good") for d in allurldata]  # all labels
    corpus = [d[0] for d in allurldata]  # all urls corresponding to a label (either good or bad)

    for classifier_key in list(classifiers):
        classifier = classifiers[classifier_key]["classifier"]
        classifiers[classifier_key]["vectorizer"] = TfidfVectorizer(
            tokenizer=getTokens)  # get a vector for each url but use our customized tokenizer
        x = classifiers[classifier_key]["vectorizer"].fit_transform(corpus)  # get the x vector

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,
                                                            random_state=42)  # split into training and testing set 80/20 ratio

        classifier.fit(x_train, y_train)

        score = classifier.score(x_test, y_test)
        accuracy = accuracy_score(y, classifier.predict(x))
        mean_absolute = mean_absolute_error(y, classifier.predict(x))

        print('Classifier : %s --- Score : %f --- Accuracy score : %f --- Mean absolute error : %f'
              % (classifier_key, score, accuracy, mean_absolute))  # pring the score. It comes out to be 98%


@app.route('/<classifier_type>/<path:path>')
def show_index(classifier_type, path):
    x_predict = classifiers[classifier_type]["vectorizer"].transform([str(path)])
    y_predict = classifiers[classifier_type]["classifier"].predict(x_predict)
    return '''You asked for "%s" AI output: %s Entropy: %s ''' % (path, str(y_predict), str(entropy(path)))


port = os.getenv('VCAP_APP_PORT', 5000)
if __name__ == "__main__":
    train_classifier()
    app.run(host='localhost', port=int(port), debug=False)
