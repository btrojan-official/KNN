import time
start_time = time.time()

import torch
import h5py

from KNN import KNN 
from load_data import load_mnist_data
from load_data import load_cifar_data

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

knn = KNN(k=3, metric="mahalanobis", device=device)

for i in range(10):

    X_train, y_train, X_test, y_test, covariances = load_cifar_data(state=i)

    knn.fit(X_train, y_train)
    # knn.replace_examples_with_mean()
    # knn.covMatrices = covariances.float().to(device)

    predictions = knn.predict(X_test)

    accuracy = torch.sum((y_test.flatten().to(device)==predictions).int())/X_test.shape[0]
    print(f"Accuracy: {accuracy.item()*100} MY")


end_time = time.time()
elapsed_time = end_time - start_time

print(f"Elapsed time: {elapsed_time} seconds")
