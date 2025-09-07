# https://scikit-learn.org/stable/api/sklearn.neighbors.html

from sklearn.datasets import load_iris
import numpy as np
from sklearn.model_selection import train_test_split
#import std scaler
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import RadiusNeighborsClassifier


def load_iris_dataset(): # -> None:
    iris = load_iris()
    return (iris.data, iris.target)

def scale_columns(traindata, testdata): # -> None:
    scaler = StandardScaler()
    scaler.fit(traindata)
    traindata[:] = scaler.transform(traindata)
    testdata[:] = scaler.transform(testdata)
    return traindata, testdata

def knn_classify(X_train, X_test, y_train, y_test): # -> None:

    print("------------------------------")

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    print(f"KNN Train score without weight:", knn.score(X_train, y_train))
    print(f"KNN Test score without weight:", knn.score(X_test, y_test))
    print(f"KNN Test Probability of Predictions:\n", knn.predict_proba(X_test))

    print("------------------------------")

    neighbors = [3, 5, 7]
    ps = [1, 2, 10]
    for ni in neighbors:
        for pi in ps:
            knn = KNeighborsClassifier(n_neighbors=ni, weights='distance', p=pi)
            knn.fit(X_train, y_train)
            print(f"KNN Train score (n={ni}, p={pi}):", knn.score(X_train, y_train))
            print(f"KNN Test score (n={ni}, p={pi}):", knn.score(X_test, y_test))

            print("------------------------------")

def rnn_classify(X_train, X_test, y_train, y_test): # -> None:

    print("------------------------------")

    rnn = RadiusNeighborsClassifier(radius=1.)
    rnn.fit(X_train, y_train)
    print("RNN Train score:", rnn.score(X_train, y_train))
    print("RNN Test score:", rnn.score(X_test, y_test))  
    print(f"RNN Test Probability of Predictions:\n", rnn.predict_proba(X_test))

    print("------------------------------")

def main() -> None:
    a, b = load_iris_dataset()
    X_train, X_test, y_train, y_test = train_test_split(a, b, test_size=0.2, random_state=42)
    X_train, X_test = scale_columns(X_train, X_test)
    knn_classify(X_train, X_test, y_train, y_test)
    rnn_classify(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()