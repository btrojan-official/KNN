print("Importing libraries...")

import torch

from KNN import KNN 
from load_data import load_mnist_data

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

print("Loading data...")
X_train, y_train, X_test, y_test = load_mnist_data()

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")


knn = KNN(k=3, metric="euclidean", device=device)

train_examples = 12000
test_examples = 1000

print("Training...")
knn.fit(X_train[:train_examples], y_train[:train_examples])

print("Predicting...")
predictions = knn.predict(X_test[:test_examples])

print("Calculating accuracy...")
accuracy = torch.sum((y_test[:test_examples].flatten().to(device)==predictions).int())/X_test[:test_examples].shape[0]
print(round(accuracy.item(),4))

