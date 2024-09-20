import time
start_time = time.time()

import torch
import h5py

from KNN import KNN 
from load_data import load_mnist_data
from load_data import load_resnet_data
from load_data import load_vit_data

import torch.nn as nn

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"DEVICE = {device}")

knn = KNN(k=1, metric="mahalanobis", weight="uniform", device=device)
knn.apply_tukeys_transformation = True
knn.lambda_hyperparameter = 1
knn.kmeans = 75
knn.l1 = 5
knn.l2 = 0

for i in range(10):

    X_train, y_train, X_test, y_test, covariances = load_vit_data(state=i)

    knn.fit(X_train, y_train)
    # knn.replace_examples_with_mean()

    # if len(covariances.size()) > 2:
    #     covariances = covariances.reshape(-1, covariances.shape[1])
    # knn.covMatrices = covariances.float().to(device)
    
    predictions = knn.predict(X_test)

    accuracy = torch.sum((y_test.flatten().to(device)==predictions).int()).float() / X_test.shape[0] * 100
    print(f"Accuracy: {accuracy.item()} MY")


# X_train, y_train, X_test, y_test, covariances = load_resnet_data(state=9)

# # knn.covMatrices = covariances.float().to(device)

# predictions = knn.predict(X_test)

# accuracy = torch.sum((y_test.flatten().to(device)==predictions).int()).float()  / X_test.shape[0]
# print(f"Accuracy: {accuracy.item()} MY")


end_time = time.time()
elapsed_time = end_time - start_time

print(f"Elapsed time: {elapsed_time} seconds")
