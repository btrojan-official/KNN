import time
start_time = time.time()

import torch

from KNN import KNN 
from load_data import load_mnist_data
from load_data import load_vit_data

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

_k = [1] # , 3, 4, 5, 7
_lambda = [5] # 0, 0.5, 1, 2, 2.5, 
_tukey = [1]
_means = [100] # 10, 25, 50, 100
_means_seed = [1]

_weights = ["uniform"] # , "distance"

k_values = []
l1_values = []
l2_values = []
tukey_values = []
means_values = []
means_seed_values = []
accuracy_values = []

with open("./results/results.txt", "a", encoding="utf-8") as f:
    test_id = 1
    test_num = len(_weights) * len(_k) * len(_lambda) * len(_lambda) * len(_tukey) * len(_means) * len(_means_seed)
    
    for weight in _weights:
        for k in _k:
            for l1 in _lambda:
                for l2 in _lambda:
                    for tukey_lambda in _tukey:
                        for kmeans in _means:
                            for kmeans_seed in _means_seed:
                                try:
                                    knn = KNN(k=k, weight=weight, metric="mahalanobis", device=device)
                                    knn.apply_tukeys_transformation = True
                                    knn.l1 = l1
                                    knn.l2 = 0
                                    knn.lambda_hyperparameter = tukey_lambda
                                    knn.kmeans = kmeans
                                    knn.kmeans_seed = kmeans_seed

                                    for i in range(10):
                                        X_train, y_train, X_test, y_test, covariances = load_vit_data(state=i)
                                        knn.fit(X_train, y_train)

                                    _, _, X_test, y_test, covariances = load_vit_data(state=9)
                                    predictions = knn.predict(X_test)

                                    message = f"{test_id}/{test_num} w = {weight}, k = {k}, l1 = {knn.l1}, l2 = {knn.l2}, tukey = {knn.apply_tukeys_transformation}, tukey_lambda = {knn.lambda_hyperparameter}, seed: {kmeans_seed}, kmeans: {kmeans}, accuracy = {torch.sum((y_test.flatten().to(device)==predictions).int()).double() / X_test.shape[0] * 100}"
                                
                                    k_values.append(k)
                                    l1_values.append(l1)
                                    l2_values.append(l2)
                                    tukey_values.append(tukey_lambda)
                                    means_values.append(kmeans)
                                    means_seed_values.append(kmeans_seed)
                                    accuracy_values.append((torch.sum((y_test.flatten().to(device)==predictions).int()).double() / X_test.shape[0] * 100).item())
                                
                                    test_id += 1
                                except:
                                    print()
                                    message = f"{test_id}/{test_num} ERROR w = {weight}, k = {k}, l1 = {knn.l1}, l2 = {knn.l2}, tukey = {knn.apply_tukeys_transformation}, tukey_lambda = {knn.lambda_hyperparameter}, seed: {kmeans_seed}, kmeans: {kmeans}, accuracy = ERROR"
                                    k_values.append(k)
                                    l1_values.append(l1)
                                    l2_values.append(l2)
                                    tukey_values.append(tukey_lambda)
                                    means_values.append(kmeans)
                                    means_seed_values.append(kmeans_seed)
                                    accuracy_values.append(-1)

                                    test_id += 1

                                f.write(message + "\n")
                                print(message)

        f.write("\n")
        print("\n")

with open("./results/parameters.txt", "a", encoding="utf-8") as f2:
    f2.write(f"w:  {_weights}\nk:  {k_values}\nl1: {l1_values}\nl2: {l2_values}\ntk: {tukey_values}\nkm: {means_values}\nac: {[val for val in accuracy_values]}\nks: {means_seed_values}")

end_time = time.time()
elapsed_time = end_time - start_time

print(f"Elapsed time: {elapsed_time} seconds")
