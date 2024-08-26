import torch

class KNN:
    def __init__(self, k=3, metric="euclidean", device=torch.device("cpu")):

        self.metrices = ["euclidean", "mahalanobis"]

        self.k = k
        if metric in self.metrices:
            self.metric = metric
        else:
            raise ValueError(f"Metric should be one of {self.metrices}")
        self.device = device
        
        self.X_train = None
        self.y_train = None

        self.covMatrices = None

    def to(self, device):

        self.device = device
        if self.X_train is not None and self.y_train is not None:
            self.X_train = self.X_train.to(device)
            self.y_train = self.y_train.to(device)

    def fit(self, X_train, y_train):

        if self.X_train is None or self.y_train is None:
            self.X_train = X_train.float().to(self.device)
            self.y_train = y_train.to(self.device)

            if self.metric == "mahalanobis":
                self.covMatrices = self._calc_covariances(X_train, y_train)
        else:
            self.X_train = torch.cat((self.X_train, X_train.float().to(self.device)))
            self.y_train = torch.cat((self.y_train, y_train.to(self.device)))

            if self.metric == "mahalanobis":
                self.covMatrices = torch.cat((self.covMatrices, self._calc_covariances(X_train, y_train))).to(self.device)

    
    def predict(self, X_test):
        X_test = X_test.float().to(self.device)

        return self._predict(X_test)
    
    def _predict(self, X_test):

        if self.metric == "euclidean":
            distances = self._euclidean(X_test)
        elif self.metric == "mahalanobis":
            distances = self._mahalanobis(X_test)

        _, knn_indices = torch.topk(distances, self.k, largest=False, dim=1, sorted=True)

        nearest_neighbours_matrix = self.y_train[knn_indices].squeeze()

        if self.k == 1:
            return nearest_neighbours_matrix

        if len(nearest_neighbours_matrix.size()) < 2:
            nearest_neighbours_matrix = nearest_neighbours_matrix.unsqueeze(0)

        batch_size, _ = nearest_neighbours_matrix.shape

        number_of_classes = torch.max(self.y_train) + 1

        counts = torch.zeros(batch_size, number_of_classes, dtype=torch.int).to(self.device)

        counts.scatter_add_(dim=1, index=nearest_neighbours_matrix, src=torch.ones_like(nearest_neighbours_matrix, dtype=torch.int))

        most_frequent = torch.argmax(counts, dim=1)

        return most_frequent
    
    def _euclidean(self, X_test, training_batch_size=10000):
        
        X_test = X_test.clone().to(self.device)
        test_squared_norms = torch.sum(X_test ** 2, dim=1).unsqueeze(1)

        for i in range(0, self.X_train.shape[0], training_batch_size): # nie wiem czy przy batch_size które nie jest dzielnikiem X_train.shape[0] weźmie wszystkie przykłady pod uwagę
            X_train_batch = self.X_train[i:i + training_batch_size, :].clone().to(self.device)

            train_squared_norms = torch.sum(X_train_batch ** 2, dim=1).unsqueeze(0)

            dot_product = torch.mm(X_test, X_train_batch.t())

            dists_squared = test_squared_norms + train_squared_norms - 2 * dot_product

            dists_squared = torch.clamp(dists_squared, min=0.0)

            if i == 0:
                dists = torch.sqrt(dists_squared)
            else:
                dists = torch.cat((dists, torch.sqrt(dists_squared)), dim=1)

        return dists

    def _mahalanobis(self, X_test):
        X_test = X_test.to(self.device)

        mahalanobis_distances = []

        for i,train_point in enumerate(self.X_train):
            f_num = self.covMatrices.shape[1]

            cov_inv = torch.inverse(self.covMatrices[self.y_train[i]*f_num:self.y_train[i]*f_num + f_num, :]).to(self.device)

            diff = (train_point - X_test).to(self.device)  # Broadcasting to compute difference with all training points
            
            mahalanobis_distance = torch.sqrt(torch.sum(diff @ cov_inv * diff, dim=1))
            mahalanobis_distances.append(mahalanobis_distance)

            # if (i+1) % 1000 == 0: print(f"{i+1} / {self.X_train.shape[0]}")

        return torch.stack(mahalanobis_distances).T.to(self.device)

    def _calc_covariances(self, X_train, y_train):
        X_train = X_train.float().to(self.device)
        y_train = y_train.to(self.device)

        uniqes = torch.unique(y_train, sorted=True).to(self.device)

        for i in uniqes:
            cov = self._calc_single_covariance(X_train, y_train, i)

            cov = self.matrix_shrinkage(cov, 1, 1)
            cov = self.normalize_covariance_matrix(cov)

            if i == uniqes[0]:
                covariances = cov
            else:
                covariances = torch.cat((covariances, cov))
        
        return covariances

    def _get_single_class_examples(self, X_train, y_train, class_number):
        y_train = y_train.view(-1).to(self.device)

        indices = (y_train == class_number).nonzero(as_tuple=True)[0].to(self.device)

        return X_train[indices]
    
    def _calc_single_covariance(self, X_train, y_train, class_number):
        single_class_examples = self._get_single_class_examples(X_train, y_train, class_number)
        
        mean = torch.mean(single_class_examples, dim=0).to(self.device)
        centered_data = single_class_examples - mean

        return (torch.mm(centered_data.t(),centered_data) / (single_class_examples.shape[0] - 1)).to(self.device)
    
    def matrix_shrinkage(self, cov_matrix, gamma1=1, gamma2=1):

        assert cov_matrix.shape[0] == cov_matrix.shape[1], "Covariance matrix must be square"
        
        I = torch.eye(cov_matrix.shape[0], device=cov_matrix.device)
        
        V1 = torch.mean(torch.diag(cov_matrix)).to(self.device)
        
        off_diagonal_elements = (cov_matrix * (1 - I)).to(self.device)
        V2 = (torch.sum(off_diagonal_elements) / (cov_matrix.shape[0] * (cov_matrix.shape[0] - 1))).to(self.device)

        shrinkaged_cov_matrix = cov_matrix + gamma1 * V1 * I + gamma2 * V2 * (1 - I)

        return shrinkaged_cov_matrix.to(self.device)
    
    def normalize_covariance_matrix(self, cov_matrix):

        diag_elements = torch.sqrt(torch.diag(cov_matrix))
        
        outer_diag = torch.outer(diag_elements, diag_elements)

        normalized_cov_matrix = cov_matrix / outer_diag
        
        return normalized_cov_matrix
    
    def replace_examples_with_mean(self):

        self.k = 1

        means = []
        labels = []

        for i in torch.unique(self.y_train, sorted=True):

            single_class_examples = self._get_single_class_examples(self.X_train, self.y_train, i)
            mean = torch.mean(single_class_examples, dim=0).unsqueeze(0)

            means.append(mean)
            labels.append(i)

        self.X_train = torch.cat(means).to(self.device)
        self.y_train = torch.tensor(labels).to(self.device)

