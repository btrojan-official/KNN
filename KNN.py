import torch
from collections import Counter

class KNN:
    def __init__(self, k=3, metric="euclidean", device=torch.device("cpu")):
        """
        Inicjalizuje klasę KNN z parametrem k, który określa liczbę najbliższych sąsiadów do wzięcia pod uwagę
        oraz metryką odległosności (jedną z ["euclidean", "mahalanobis"]).
        
        Parametry:
        k (int): Liczba najbliższych sąsiadów.
        metric (str): Metryka odległosności.
        """
        self.metrices = ["euclidean", "mahalanobis"]

        self.k = k
        if metric in self.metrices:
            self.metric = metric
        else:
            raise ValueError(f"Metric should be one of {self.metrices}")
        self.device = device
        
        self.X_train = None
        self.y_train = None
    def to(self, device):
        """
        Moves the KNN model to a specified device.

        Parameters:
        device (torch.device): The device to which the model should be moved.
        """

        self.device = device
        if self.X_train is not None and self.y_train is not None:
            self.X_train = self.X_train.to(device)
            self.y_train = self.y_train.to(device)

    def fit(self, X_train, y_train):
        """
        Trenuje model KNN, zapamiętując dane treningowe.

        Parametry:
        X_train (torch.Tensor): Dane treningowe.
        y_train (torch.Tensor): Etykiety danych treningowych.
        """

        if self.X_train is None or self.y_train is None:
            self.X_train = X_train.float().to(self.device)
            self.y_train = y_train.float().to(self.device)
        else:
            pass
    
    def predict(self, X_test):
        """
        Przewiduje etykiety dla danych testowych na podstawie k najbliższych sąsiadów.

        Parametry:
        X_test (torch.Tensor): Dane testowe.

        Zwraca:
        torch.Tensor: Przewidywane etykiety dla danych testowych.
        """
        X_test = X_test.float().to(self.device)

        return self._predict(X_test)
    
    def _predict(self, X_test):
        """
        Przewiduje etykietę dla pojedynczego przykładu X_test na podstawie k najbliższych sąsiadów.

        Parametry:
        X_test (torch.Tensor): Tensor danych testowych.

        Zwraca:
        torch.Tensor: Przewidywana etykieta dla podanych przykładów.
        """
        # Oblicz odległość między x a wszystkimi przykładami w zbiorze treningowym
        distances = [torch.linalg.norm(X_test - x_train) for x_train in self.X_train]
        
        # Znajdź k najbliższych sąsiadów
        k_indices = torch.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        # Zwróć najczęściej występującą etykietę (dla klasyfikacji)
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
