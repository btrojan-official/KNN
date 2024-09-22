import torch
from sklearn.cluster import KMeans

class KNN:
    def __init__(self, k=3, metric="euclidean", weight="uniform", device=torch.device("cpu")):

        self.metrices = ["euclidean", "mahalanobis"]
        self.weights = ["uniform", "distance"]

        self.k = k
        if metric in self.metrices:
            self.metric = metric
        else:
            raise ValueError(f"Metric should be one of {self.metrices}")
        
        if weight in self.weights:
            self.weight = weight
        else:
            raise ValueError(f"Weight should be one of {self.weights}")

        self.device = device
        
        self.X_train = None
        self.y_train = None

        self.l1 = 1
        self.l2 = 0

        self.lambda_hyperparameter = 0.5

        self.apply_tukeys_transformation = False

        self.kmeans = 0
        self.kmeans_seed = 1

        self.covMatrices = None

    def to(self, device):

        self.device = device
        if self.X_train is not None and self.y_train is not None:
            self.X_train = self.X_train.to(device)
            self.y_train = self.y_train.to(device)

    def fit(self, X_train, y_train):

        if self.X_train is None or self.y_train is None:
            if self.metric == "mahalanobis":
                if self.apply_tukeys_transformation:
                    self.covMatrices = self._calc_covariances(self._tukeys_transformation(X_train), y_train)
                else:
                    self.covMatrices = self._calc_covariances(X_train, y_train)
        else:
            if self.metric == "mahalanobis":
                if self.apply_tukeys_transformation:
                    self.covMatrices = torch.cat((self.covMatrices, self._calc_covariances(self._tukeys_transformation(X_train), y_train))).to(self.device)
                else:
                    self.covMatrices = torch.cat((self.covMatrices, self._calc_covariances(X_train, y_train))).to(self.device)
        
        if self.kmeans > 0:
            uniqes = torch.unique(y_train, sorted=True).to(self.device)

            for i in uniqes:
                single_class_examples = self._get_single_class_examples(X_train.float().to(self.device), y_train.float().to(self.device), i)
                if i == torch.min(uniqes): 
                    new_X_train = self._kmeans(single_class_examples).to(self.device)
                    new_y_train = torch.full((self.kmeans,), i.item()).to(self.device)
                else:
                    new_X_train = torch.cat((new_X_train, self._kmeans(single_class_examples).to(self.device)))
                    new_y_train = torch.cat((new_y_train, torch.full((self.kmeans,), i.item()).to(self.device)))

            X_train = new_X_train
            y_train = new_y_train

        if self.X_train is None or self.y_train is None:
            self.X_train = X_train.float().to(self.device)
            self.y_train = y_train.to(self.device)
        else:
            self.X_train = torch.cat((self.X_train, X_train.float().to(self.device)))
            self.y_train = torch.cat((self.y_train, y_train.to(self.device)))
    
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

        counts = torch.zeros(batch_size, number_of_classes, dtype=torch.float).to(self.device)

        if self.weight == "uniform": weights_matrix = torch.ones_like(nearest_neighbours_matrix, dtype=torch.float).to(self.device)
        elif self.weight == "distance": weights_matrix = 1 / torch.gather(distances, 1, knn_indices).to(self.device)
        else: raise ValueError(f"Weight should be one of {self.weights}")

        counts.scatter_add_(dim=1, index=nearest_neighbours_matrix, src=(weights_matrix))            

        most_frequent = torch.argmax(counts, dim=1)

        def is_draw(tensor):
            sorted_tensor, _ = tensor.sort(dim=0, descending=True)

            max_values = sorted_tensor[0]
            second_max_values = sorted_tensor[1]
            return max_values == second_max_values

        for i,line in enumerate(counts):
            if is_draw(line):
                most_frequent[i] = nearest_neighbours_matrix[i][0]

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

    def _mahalanobis(self, X_test, batch_size=16):
        
        X_test = X_test.to(self.device)

        if self.apply_tukeys_transformation:
            X_test = self._tukeys_transformation(X_test)
            self.X_train = self._tukeys_transformation(self.X_train)

        f_num = self.covMatrices.shape[1]
        num_classes = self.covMatrices.shape[0] // f_num

        cov_inv_list = []
        for i in range(num_classes):  
            cov_inv = torch.inverse(self.covMatrices[i * f_num:(i + 1) * f_num, :]).to(self.device)
            cov_inv_list.append(cov_inv)
        cov_inv_stack = torch.stack(cov_inv_list)

        mahalanobis_distances = []

        for start_idx in range(0, self.X_train.shape[0], batch_size):
            end_idx = min(start_idx + batch_size, self.X_train.shape[0])
            X_train_batch = self.X_train[start_idx:end_idx].to(self.device)
            cov_inv_per_batch = cov_inv_stack[self.y_train[start_idx:end_idx]]
            X_test_exp = X_test.unsqueeze(0).repeat(end_idx - start_idx, 1, 1)
            diff = (X_test_exp - X_train_batch.unsqueeze(1))# (torch.nn.functional.normalize(X_train_batch.unsqueeze(1), p=2, dim=-1) - torch.nn.functional.normalize(X_test_exp, p=2, dim=-1)).to(self.device)
            batch_distances = torch.sqrt(torch.sum(diff @ cov_inv_per_batch * diff, dim=2))
            mahalanobis_distances.append(batch_distances)

        return torch.cat(mahalanobis_distances, dim=0).T.to(self.device)


    def _calc_covariances(self, X_train, y_train):
        X_train = X_train.float().to(self.device)
        y_train = y_train.to(self.device)

        uniqes = torch.unique(y_train, sorted=True).to(self.device)

        for i in uniqes:
            cov = self._calc_single_covariance(X_train, y_train, i)

            cov = self.matrix_shrinkage(cov, self.l1, self.l2)#self.matrix_shrinkage(, 1, 0)
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
        
        # mean = torch.mean(single_class_examples, dim=0).to(self.device)
        # centered_data = single_class_examples - mean

        # return (torch.mm(centered_data.t(),centered_data) / (single_class_examples.shape[0] - 1)).to(self.device)
        return torch.cov(single_class_examples.T).to(self.device)
    
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
    
    def _tukeys_transformation(self, x: torch.Tensor) -> torch.Tensor:
        if self.lambda_hyperparameter != 0:
            return x**self.lambda_hyperparameter
        else:
            return torch.log(x)
    
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

    def _kmeans(self, X_train):
        kmeans = KMeans(n_clusters=self.kmeans, random_state=self.kmeans_seed)
        kmeans.fit(X_train.cpu().numpy())

        cluster_centers = kmeans.cluster_centers_

        cluster_centers_tensor = torch.tensor(cluster_centers, dtype=X_train.dtype)

        return cluster_centers_tensor
