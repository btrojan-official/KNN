import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from load_data import load_resnet_data
from sklearn.manifold import TSNE

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"DEVICE = {device}")

class Classifier(torch.nn.Module):
    def __init__(self, input, hidden1, hidden2, output):
        super(Classifier, self).__init__()

        self.model = torch.nn.Sequential(
            torch.nn.Linear(input, hidden1),
            torch.nn.Sigmoid(),
            torch.nn.Linear(hidden1, hidden2),
            torch.nn.Sigmoid(),
            torch.nn.Linear(hidden2, output),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)
    


X_train_full, y_train_full, _, _, _ = load_resnet_data(0)
for i in range(1,6):
    X_train, y_train, _, _, _ = load_resnet_data(i)

    X_train_full = torch.cat((X_train_full, X_train))
    y_train_full = torch.cat((y_train_full, y_train))

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

_, _, X_test, y_test, _ = load_resnet_data(state=5)

print("Fitting to tsne")
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_train_full = torch.tensor(tsne.fit_transform(np.array(X_train_full)))

train_set = CustomDataset(X_train_full.to(device), y_train_full.to(device))
test_set = CustomDataset(X_test.to(device), y_test.to(device))

train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
test_loader = DataLoader(test_set, batch_size=128, shuffle=False)

lr = 0.01
epoch_num = 50

classifier = Classifier(X_train_full.shape[1], 24, 56, len(np.unique(y_train))).to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=classifier.parameters(), lr=lr)

for epoch in range(epoch_num):
    loss_train = []
    loss_test = []

    for imgs, labels in train_loader:
        opts = classifier(imgs)

        optimizer.zero_grad()
        loss = criterion(opts, labels)
        loss.backward()
        optimizer.step()

        loss_train.append(loss.item())

    for imgs, labels in train_loader:
        opts = classifier(imgs)
        loss = torch.argmax(opts, 1) == labels

        loss_test.append(sum(loss))

    print(f"{epoch+1}/{epoch_num} Loss: {sum(loss_train)/len(loss_train)} Test acc: {sum(loss_test)/len(train_set)}")
