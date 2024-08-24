print("Importing libraries...")
import torch


from KNN import KNN 
from load_data import load_mnist_data

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

print("Loading data...")
X_train = torch.tensor([[1,2],[2,3],[3,5]])
y_train = torch.tensor([[0],[0],[0]])

X_test = torch.tensor([[2,4],[2,4]])
y_test = torch.tensor([[2]])

knn = KNN(k=1, metric="mahalanobis", device=device)

print("Training...")
knn.fit(X_train, y_train)

print("Predicting...")
predictions = knn.predict(X_test)

print("Calculating accuracy...")
accuracy = torch.sum((y_test.flatten().to(device)==predictions).int())/X_test.shape[0]
print(round(accuracy.item(),4))

