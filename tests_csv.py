import time
import csv
import torch

from KNN import KNN 
from load_data import load_mnist_data, load_resnet_data, load_vit_data

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

_k = [1, 3, 4]
_lambda = [0, 0.5, 1, 2, 2.5, 5, 10]
_tukey = [1]
_means = [10, 25, 50, 100]
_means_seed = [200, 300, 400]

_weights = ["uniform", "distance"]

start_time = time.time()

with open("./results/results.csv", "a", newline="", encoding="utf-8") as f:
    csv_writer = csv.writer(f)
    
    f.seek(0, 2) 
    if f.tell() == 0:
        csv_writer.writerow(["test_id", "weight", "k", "l1", "l2", "tukey", "tukey_lambda", "seed", "kmeans", "accuracy"])
    
    test_id = 1
    test_num = len(_weights) * len(_k) * len(_lambda) * len(_lambda) * len(_tukey) * len(_means) * len(_means_seed)
    if 0 in _lambda:
        test_num = int(test_num - (test_num / len(_lambda)))
        
    for weight in _weights:
        for k in _k:
            for l1 in _lambda:
                for l2 in _lambda:
                    for tukey_lambda in _tukey:
                        for kmeans in _means:
                            for kmeans_seed in _means_seed:
                                if l1 == 0:
                                    continue
                                try:
                                    knn = KNN(k=k, weight=weight, metric="mahalanobis", device=device)
                                    knn.apply_tukeys_transformation = True
                                    knn.l1 = l1
                                    knn.l2 = l2
                                    knn.lambda_hyperparameter = tukey_lambda
                                    knn.kmeans = kmeans
                                    knn.kmeans_seed = kmeans_seed

                                    for i in range(10):
                                        X_train, y_train, X_test, y_test, covariances = load_vit_data(state=i)
                                        knn.fit(X_train, y_train)

                                    _, _, X_test, y_test, covariances = load_vit_data(state=9)
                                    predictions = knn.predict(X_test)

                                    accuracy = (torch.sum((y_test.flatten().to(device) == predictions).int()).double() / X_test.shape[0] * 100).item()
                                    message = f"{test_id}/{test_num} w = {weight}, k = {k}, l1 = {knn.l1}, l2 = {knn.l2}, tukey = {knn.apply_tukeys_transformation}, tukey_lambda = {knn.lambda_hyperparameter}, seed: {kmeans_seed}, kmeans: {kmeans}, accuracy = {accuracy}"

                                    csv_writer.writerow([test_id, weight, k, l1, l2, knn.apply_tukeys_transformation, tukey_lambda, kmeans_seed, kmeans, accuracy])
                                    test_id += 1

                                except Exception as e:
                                    print(f"Error occurred: {e}")
                                    message = f"{test_id}/{test_num} ERROR w = {weight}, k = {k}, l1 = {knn.l1}, l2 = {knn.l2}, tukey = {knn.apply_tukeys_transformation}, tukey_lambda = {knn.lambda_hyperparameter}, seed: {kmeans_seed}, kmeans: {kmeans}, accuracy = ERROR"
                                    csv_writer.writerow([test_id, weight, k, l1, l2, knn.apply_tukeys_transformation, tukey_lambda, kmeans_seed, kmeans, "ERROR"])
                                    test_id += 1

                                print(message)

    print("\n")

end_time = time.time()
elapsed_time = end_time - start_time

print(f"Elapsed time: {elapsed_time} seconds")
