import time
start_time = time.time()

import torch

from KNN import KNN 
from load_data import load_mnist_data
from load_data import load_resnet_data

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

_k = [1, 3, 10, 50]
_lambda = [0, 0.5, 1, 5, 50]

_weights = ["uniform", "distance"]

k_values = []
l1_values = []
l2_values = []
accuracy_values = []

with open("results.txt", "a", encoding="utf-8") as f:
    test_id = 1
    test_num = len(_weights) * len(_k) * len(_lambda) * len(_lambda)
    
    for weight in _weights:
        for k in _k:
            for l1 in _lambda:
                for l2 in _lambda:
                    try:
                        knn = KNN(k=k, weight=weight, metric="mahalanobis", device=device)
                        knn.apply_tukeys_transformation = True
                        knn.l1 = l1
                        knn.l2 = l2

                        for i in range(6):
                            X_train, y_train, X_test, y_test, covariances = load_resnet_data(state=i)
                            knn.fit(X_train, y_train)

                        _, _, X_test, y_test, covariances = load_resnet_data(state=5)
                        predictions = knn.predict(X_test)

                        message = f"{test_id}/{test_num} w = {weight}, k = {k}, l1 = {knn.l1}, l2 = {knn.l2}, tukey = {knn.apply_tukeys_transformation}, tukey_lambda = {knn.lambda_hyperparameter}, accuracy = {torch.sum((y_test.flatten().to(device)==predictions).int()).double() / X_test.shape[0] * 100}"
                    
                        k_values.append(k)
                        l1_values.append(l1)
                        l2_values.append(l2)
                        accuracy_values.append(torch.sum((y_test.flatten().to(device)==predictions).int()).double() / X_test.shape[0] * 100)
                    
                        test_id += 1
                    except:
                        print()
                        message = f"{test_id}/{test_num} ERROR w = {weight}, k = {k}, l1 = {knn.l1}, l2 = {knn.l2}, tukey = {knn.apply_tukeys_transformation}, tukey_lambda = {knn.lambda_hyperparameter}, accuracy = ERROR"
                        k_values.append(k)
                        l1_values.append(l1)
                        l2_values.append(l2)
                        accuracy_values.append(-1)

                        test_id += 1

                    f.write(message + "\n")
                    print(message)

        f.write("\n")
        print("\n")

with open("parameters.txt", "a", encoding="utf-8") as f2:
    f2.write(f"w:  {_weights}\nk:  {k_values}\nl1: {l1_values}\nl2: {l2_values}\nac: {[val.cpu().item() for val in accuracy_values]}\n\n")

end_time = time.time()
elapsed_time = end_time - start_time

print(f"Elapsed time: {elapsed_time} seconds")
