# print("Importing libraries...")

import torch
import h5py

from KNN import KNN 
from load_data import load_mnist_data
from load_data import load_cifar_data

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

knn = KNN(k=3, metric="mahalanobis", device=device)

for i in range(10):
    # print(f"{i} Loading data...")
    X_train, y_train, X_test, y_test = load_cifar_data(state=i)

    # print(f"{i} Training...")
    knn.fit(X_train, y_train)

    # print(f"{i} Predicting...")
    predictions = knn.predict(X_test)

    # print(f"{i} Calculating accuracy...")
    accuracy = torch.sum((y_test.flatten().to(device)==predictions).int())/X_test.shape[0]
    print(f"KNN mahalanobis: {round(accuracy.item(),4)}")

