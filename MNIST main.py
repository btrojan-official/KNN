print("Importing libraries...")

import torch

from KNN import KNN 
from load_data import load_mnist_data

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

print("Loading data...")
X_train, y_train, X_test, y_test = load_mnist_data()

knn = KNN(k=10, metric="mahalanobis", weight="distance", device=device)

train_examples = 6000
test_examples = 10000

print("Training...")
# print("!!!Replacing examples with mean...")
knn.fit(X_train[:train_examples], y_train[:train_examples]) 
# knn.replace_examples_with_mean()

print("Predicting...")
predictions = knn.predict(X_test[:test_examples])

print("Calculating accuracy...")
accuracy = torch.sum((y_test[:test_examples].flatten().to(device)==predictions).int())/X_test[:test_examples].shape[0]
print(round(accuracy.item(),4))

