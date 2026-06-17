import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np


class IndexedTensorDataset(torch.utils.data.Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = np.array(targets, dtype=np.int64)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.data[idx], int(self.targets[idx])


class DataLoaderManager:
    def __init__(self, config):
        self.config = config
        self.train_dataset = None
        self.test_dataset = None
        self.user_groups = None

    def load_data(self, dataset_name='CIFAR10'):
        if dataset_name in ('CIFAR10', 'CIFAR100'):
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            dataset_cls = datasets.CIFAR10 if dataset_name == 'CIFAR10' else datasets.CIFAR100
            self.train_dataset = dataset_cls(self.config["data_path"], train=True, download=True, transform=transform_train)
            self.test_dataset = dataset_cls(self.config["data_path"], train=False, download=True, transform=transform_test)
        elif dataset_name == 'MNIST':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            self.train_dataset = datasets.MNIST(self.config["data_path"], train=True, download=True, transform=transform)
            self.test_dataset = datasets.MNIST(self.config["data_path"], train=False, download=True, transform=transform)
        elif dataset_name == 'EMNIST':
            emnist_split = self.config.get("emnist_split", "byclass")
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ])
            self.train_dataset = datasets.EMNIST(
                self.config["data_path"],
                split=emnist_split,
                train=True,
                download=True,
                transform=transform,
            )
            self.test_dataset = datasets.EMNIST(
                self.config["data_path"],
                split=emnist_split,
                train=False,
                download=True,
                transform=transform,
            )
        elif dataset_name == 'MNIST_FEDPROX':
            # Match litian96/FedProx data/mnist/generate_niid.py: normalize the
            # full 70k MNIST-original samples before client-level 90/10 splitting.
            raw_train = datasets.MNIST(
                self.config["data_path"], train=True, download=True
            )
            raw_test = datasets.MNIST(
                self.config["data_path"], train=False, download=True
            )
            full_x = np.concatenate(
                [raw_train.data.numpy(), raw_test.data.numpy()], axis=0
            ).astype(np.float32)
            full_y = np.concatenate(
                [np.array(raw_train.targets), np.array(raw_test.targets)], axis=0
            ).astype(np.int64)

            flat_x = full_x.reshape(len(full_x), -1)
            mu = flat_x.mean(axis=0)
            sigma = flat_x.std(axis=0)
            norm_x = ((flat_x - mu) / (sigma + 0.001)).reshape(-1, 1, 28, 28)

            self._fedprox_full_data = torch.from_numpy(norm_x).float()
            self._fedprox_full_targets = full_y
            self.train_dataset = IndexedTensorDataset(
                self._fedprox_full_data, self._fedprox_full_targets
            )
            self.test_dataset = self.train_dataset

        return self.train_dataset, self.test_dataset

    def partition_data(self, method='iid', alpha=0.5, similarity_frac=None, **kwargs):
        num_clients = self.config["num_clients"]
        dataset = self.train_dataset
        labels = np.array(dataset.targets)
        num_classes = len(np.unique(labels))
        dict_users = {i: np.array([], dtype='int64') for i in range(num_clients)}

        if method == 'iid':
            num_items = int(len(dataset) / num_clients)
            all_idxs = [i for i in range(len(dataset))]
            for i in range(num_clients):
                dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
                all_idxs = list(set(all_idxs) - dict_users[i])
        
        elif method == 'shards':
            idxs = np.argsort(labels)
            num_shards = 2 * num_clients
            shard_size = len(dataset) // num_shards
            shard_idxs = [i for i in range(num_shards)]
            for i in range(num_clients):
                rand_shards = np.random.choice(shard_idxs, 2, replace=False)
                shard_idxs = list(set(shard_idxs) - set(rand_shards))
                dict_users[i] = np.concatenate([idxs[s*shard_size : (s+1)*shard_size] for s in rand_shards])
        
        elif method == 'dirichlet':
            dict_users = self._partition_dirichlet_balanced(labels, num_clients, num_classes, alpha)
        elif method == 'fedprox_mnist':
            dict_users = self._partition_mnist_fedprox(
                num_clients=num_clients,
                train_frac=float(self.config.get("client_train_frac", 0.9)),
                seed=int(self.config.get("seed", 0)),
            )
        elif method == 'emnist_similarity':
            if similarity_frac is None:
                similarity_frac = float(self.config.get("similarity_frac", 0.0))
            dict_users = self._partition_emnist_similarity(
                num_clients=num_clients,
                similarity_frac=float(similarity_frac),
                seed=int(self.config.get("seed", 0)),
            )

        self.user_groups = dict_users
        return dict_users

    def _partition_emnist_similarity(self, num_clients, similarity_frac, seed=0):
        """
        SCAFFOLD paper (Section 7.1): for s% similar data, split the training
        samples into an s% IID pool and a (100-s)% label-sorted pool, then give
        each client equal-sized shards from both pools (cf. Hsu et al. 2019).

        similarity_frac in [0, 1] matches paper 's% similar' (0 = sorted non-IID).
        """
        similarity_frac = float(np.clip(similarity_frac, 0.0, 1.0))
        rng = np.random.RandomState(seed)
        labels = np.array(self.train_dataset.targets)
        num_samples = len(labels)
        per_client = num_samples // num_clients
        iid_count = int(round(per_client * similarity_frac))
        sorted_count = per_client - iid_count

        shuffled_idxs = rng.permutation(np.arange(num_samples))
        iid_total = iid_count * num_clients
        sorted_total = sorted_count * num_clients

        iid_pool = shuffled_idxs[:iid_total]
        sorted_pool = shuffled_idxs[iid_total:iid_total + sorted_total]
        sorted_pool = sorted_pool[np.argsort(labels[sorted_pool])]

        iid_shards = [
            iid_pool[i * iid_count:(i + 1) * iid_count]
            for i in range(num_clients)
        ]
        sorted_shards = [
            sorted_pool[i * sorted_count:(i + 1) * sorted_count]
            for i in range(num_clients)
        ]

        dict_users = {}
        for client_id in range(num_clients):
            client_idxs = []

            if iid_count > 0:
                client_idxs.extend(iid_shards[client_id].tolist())
            if sorted_count > 0:
                client_idxs.extend(sorted_shards[client_id].tolist())

            rng.shuffle(client_idxs)
            dict_users[client_id] = np.array(client_idxs[:per_client], dtype=np.int64)

        return dict_users

    def _partition_mnist_fedprox(self, num_clients=1000, train_frac=0.9, seed=0):
        """
        MNIST partition from FedProx paper / litian96/FedProx data/mnist/generate_niid.py:
        1000 clients, each with 2 digit classes, sample counts follow a power law.
        """
        rng = np.random.RandomState(seed)
        full_data = self._fedprox_full_data
        labels = np.array(self._fedprox_full_targets)
        class_indices = [np.where(labels == digit)[0] for digit in range(10)]

        idx = np.zeros(10, dtype=np.int64)
        client_indices = {user: [] for user in range(num_clients)}

        for user in range(num_clients):
            for j in range(2):
                digit = (user + j) % 10
                end = idx[digit] + 5
                client_indices[user].extend(class_indices[digit][idx[digit]:end].tolist())
                idx[digit] += 5

        # Match litian96/FedProx generate_niid.py power-law assignment.
        props = rng.lognormal(0, 2.0, (10, 100, 2))
        remaining = np.array(
            [len(class_indices[digit]) - 1000 for digit in range(10)],
            dtype=np.float64,
        )
        props = remaining.reshape(10, 1, 1) * props
        props /= props.sum(axis=(1, 2), keepdims=True)

        for user in range(num_clients):
            for j in range(2):
                digit = (user + j) % 10
                num_samples = int(props[digit, user // 10, j])
                end = idx[digit] + num_samples
                if end < len(class_indices[digit]):
                    client_indices[user].extend(class_indices[digit][idx[digit]:end].tolist())
                idx[digit] += num_samples

        dict_users = {}
        train_indices = []
        test_indices = []
        for user in range(num_clients):
            indices = np.array(client_indices[user], dtype=np.int64)
            rng.shuffle(indices)
            train_len = max(1, int(train_frac * len(indices)))
            train_part = indices[:train_len]
            test_part = indices[train_len:]

            start = len(train_indices)
            train_indices.extend(train_part.tolist())
            dict_users[user] = np.arange(start, start + len(train_part), dtype=np.int64)
            test_indices.extend(test_part.tolist())

        train_indices = np.array(train_indices, dtype=np.int64)
        test_indices = np.array(test_indices, dtype=np.int64)
        self.train_dataset = IndexedTensorDataset(
            full_data[train_indices], labels[train_indices]
        )
        self.test_dataset = IndexedTensorDataset(
            full_data[test_indices], labels[test_indices]
        )

        return dict_users

    def _partition_dirichlet_balanced(self, labels, num_clients, num_classes, alpha):
        client_indices = [[] for _ in range(num_clients)]

        for class_id in range(num_classes):
            class_idxs = np.where(labels == class_id)[0]
            np.random.shuffle(class_idxs)
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            split_sizes = np.random.multinomial(len(class_idxs), proportions)

            start = 0
            for client_id, split_size in enumerate(split_sizes):
                end = start + split_size
                if split_size > 0:
                    client_indices[client_id].extend(class_idxs[start:end].tolist())
                start = end

        target_size = len(labels) // num_clients
        overflow_pool = []
        deficits = []
        for client_id, indices in enumerate(client_indices):
            np.random.shuffle(indices)
            if len(indices) > target_size:
                overflow_pool.extend(indices[target_size:])
                client_indices[client_id] = indices[:target_size]
            elif len(indices) < target_size:
                deficits.append((client_id, target_size - len(indices)))

        np.random.shuffle(overflow_pool)
        cursor = 0
        for client_id, deficit in deficits:
            if deficit <= 0:
                continue
            take = overflow_pool[cursor:cursor + deficit]
            client_indices[client_id].extend(take)
            cursor += len(take)

        leftovers = overflow_pool[cursor:]
        if leftovers:
            for idx, sample_idx in enumerate(leftovers):
                client_indices[idx % num_clients].append(sample_idx)

        return {
            client_id: np.array(client_indices[client_id], dtype='int64')
            for client_id in range(num_clients)
        }
