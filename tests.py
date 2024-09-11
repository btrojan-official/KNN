import time
start_time = time.time()

import torch

from KNN import KNN 
from load_data import load_mnist_data
from load_data import load_cifar_data

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

_k = [1, 3, 5, 10, 20, 50, 100]
_lambda = [0.5, 1, 2, 5, 10, 50, 100, 150, 300, 500]

_weights = ["uniform", "distance"]

with open("results.txt", "w", encoding="utf-8") as f:
    for weight in _weights:
        for k in _k:
            try:
                knn = KNN(k=k, weight=weight, metric="mahalanobis", device=device)

                for i in range(10):
                    X_train, y_train, X_test, y_test, covariances = load_cifar_data(state=i)
                    knn.fit(X_train, y_train)

                _, _, X_test, y_test, covariances = load_cifar_data(state=9)
                predictions = knn.predict(X_test)

                message = f"w = {weight}, k = {k}, l1 = {knn.l1}, l2 = {knn.l2}, accuracy = {torch.sum((y_test.flatten().to(device)==predictions).int()).double() / X_test.shape[0] * 100}"
            except:
                message = f"ERROR w = {weight}, k = {k}, l1 = {knn.l1}, l2 = {knn.l2}, accuracy = ERROR"
            f.write(message + "\n")
            print(message)

        f.write("\n")
        print("\n")

        for l1 in _lambda:
            try:
                knn = KNN(k=3, weight=weight,metric="mahalanobis", device=device)
                knn.l1 = l1
                knn.l2 = l1

                for i in range(1):
                    X_train, y_train, X_test, y_test, covariances = load_cifar_data(state=i)
                    knn.fit(X_train, y_train)

                _, _, X_test, y_test, covariances = load_cifar_data(state=0)
                predictions = knn.predict(X_test)

                message = f"w = {weight}, k = {3}, l1 = {knn.l1}, l2 = {knn.l2}, accuracy = {torch.sum((y_test.flatten().to(device)==predictions).int()).double() / X_test.shape[0] * 100}"
            except:
                message = f"ERROR w = {weight}, k = {3}, l1 = {knn.l1}, l2 = {knn.l2}, accuracy = ERROR"
            f.write(message + "\n")
            print(message)

        f.write("\n")
        print("\n")

end_time = time.time()
elapsed_time = end_time - start_time

print(f"Elapsed time: {elapsed_time} seconds")
