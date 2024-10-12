import time
start_time = time.time()

import torch

from KNN import KNN #knn with different transformation
from load_data import load_mnist_data, load_resnet_data, load_vit_data

from torch.nn import MSELoss

import torch.nn as nn

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"DEVICE = {device}")

knn = KNN(k=1, metric="mahalanobis", weight="distance", device=device)
knn.apply_tukeys_transformation = True
knn.lambda_hyperparameter = 0.5
knn.kmeans = 1
knn.l1 = 1
knn.l2 = 1
knn.kmeans_seed = 41

for i in range(6):

    X_train, y_train, X_test, y_test, covariances = load_resnet_data(state=i, load_covariances=True)

    knn.fit(X_train, y_train)
    # knn.replace_examples_with_mean()

    if len(covariances.size()) > 2:
        covariances = covariances.reshape(-1, covariances.shape[1]).float().to(device)
    # knn.covMatrices = covariances

    print(knn.cov_mse(covariances))
    
    predictions = knn.predict(X_test)

    accuracy = torch.sum((y_test.flatten().to(device)==predictions).int()).float() / X_test.shape[0] * 100
    print(f"Accuracy: {accuracy.item()} MY")

# X_train, y_train, X_test, y_test, covariances = load_resnet_data(state=9, load_covariances=True)

# # knn.covMatrices = covariances.float().to(device)

# predictions = knn.predict(X_test)

# accuracy = torch.sum((y_test.flatten().to(device)==predictions).int()).float()  / X_test.shape[0]
# print(f"Accuracy: {accuracy.item()} MY")


end_time = time.time()
elapsed_time = end_time - start_time

print(f"Elapsed time: {elapsed_time} seconds")
