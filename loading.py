import h5py
import numpy as np

number_of_task = 9
current_file = f"./ViT_pretrained_CIFAR_100/task_{number_of_task}.hdf5"

with h5py.File(current_file, "r") as f:
    X_train = f["X_train"][:]
    y_train = f["y_train"][:]
    covariances = f["covariances"][:]
    X_test = f["X_test"][:]
    y_test = f["y_test"][:]
    test_predictions = f["test_predictions"][:]
    task_id = f["info"].attrs["task"]
    accuracy = f["info"].attrs["accuracy"]

    print(X_train.shape, y_train.shape, covariances.shape, X_test.shape, y_test.shape, test_predictions.shape, task_id, accuracy)

    print(np.unique(y_test))

