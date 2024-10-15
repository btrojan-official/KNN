import torch
from KNN_resnet import KNN 
from load_data import load_vit_data, load_resnet_data

def accuracy(y_test, predictions):
    return (torch.sum((y_test==predictions).int()) / X_test.shape[0] * 100).item()

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"DEVICE = {device}")


knn = KNN(k=1, metric="mahalanobis", weight="uniform", device=device)
knn.apply_tukeys_transformation = True
knn.lambda_hyperparameter = 0.5
knn.kmeans = 1
knn.l1 = 1
knn.l2 = 1
knn.kmeans_seed = 45

for i in range(6):

    X_train, y_train, X_test, y_test, covariances, prototypes = load_resnet_data(state=i, load_covariances=True, load_prototypes=True)
    prototypes = prototypes.to(device)

    knn.fit(X_train, y_train)

    if len(covariances.size()) > 2:
        covariances = covariances.reshape(-1, covariances.shape[1]).float().to(device)
    else:
        covariances = covariances.float().to(device)

    # knn.covMatrices = covariances
    # knn.X_train = prototypes

    predictions = knn.predict(X_test)

    print(f"Accuracy: {accuracy(y_test.flatten().to(device), predictions)} MY")

    print(f"\nCov MSE: {knn.cov_mse(covariances)}")

    print(f"Prototypes MSE: {knn.prototypes_mse(prototypes)}\n")