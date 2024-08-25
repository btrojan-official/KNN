import torch
import torchvision
import torchvision.transforms as transforms

import h5py

def load_mnist_data():
    transform = transforms.Compose([transforms.ToTensor()])
    mnist_train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    mnist_test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    flattened_images = []
    labels = []

    for image, label in mnist_test:
        flattened_image = image.flatten()
        flattened_images.append(flattened_image)
        labels.append(label)

    tensor_test_imgs = torch.stack(flattened_images)
    tensor_test_labels = torch.tensor(labels).unsqueeze(1)

    flattened_images = []
    labels = []

    for image, label in mnist_train:
        flattened_image = image.flatten()
        flattened_images.append(flattened_image)
        labels.append(label)

    tensor_train_imgs = torch.stack(flattened_images)
    tensor_train_labels = torch.tensor(labels).unsqueeze(1)

    return tensor_train_imgs, tensor_train_labels, tensor_test_imgs, tensor_test_labels

def load_cifar_data(state=0):
    current_file = f"./ViT_pretrained_CIFAR_100/task_{state}.hdf5"

    with h5py.File(current_file, "r") as f:
        X_train = f["X_train"][:]
        y_train = f["y_train"][:]
        covariances = f["covariances"][:]
        X_test = f["X_test"][:]
        y_test = f["y_test"][:]
        test_predictions = f["test_predictions"][:]
        task_id = f["info"].attrs["task"]
        accuracy = f["info"].attrs["accuracy"]

        print(f"Accuracy: {accuracy}, Task ID: {task_id}")

        return torch.tensor(X_train), torch.tensor(y_train), torch.tensor(X_test), torch.tensor(y_test), torch.tensor(covariances)
