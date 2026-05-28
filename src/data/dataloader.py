import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

class DataLoaderManager:
    def __init__(self, config):
        self.config = config
        self.train_dataset = None
        self.test_dataset = None
        self.user_groups = None

    def load_data(self, dataset_name='CIFAR10'):
        if dataset_name == 'CIFAR10':
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
            self.train_dataset = datasets.CIFAR10(self.config["data_path"], train=True, download=True, transform=transform_train)
            self.test_dataset = datasets.CIFAR10(self.config["data_path"], train=False, download=True, transform=transform_test)
        elif dataset_name == 'MNIST':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            self.train_dataset = datasets.MNIST(self.config["data_path"], train=True, download=True, transform=transform)
            self.test_dataset = datasets.MNIST(self.config["data_path"], train=False, download=True, transform=transform)
        
        return self.train_dataset, self.test_dataset

    def partition_data(self, method='iid', alpha=0.5):
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

        self.user_groups = dict_users
        return dict_users

    def _partition_dirichlet_balanced(self, labels, num_clients, num_classes, alpha):
        target_size = len(labels) // num_clients
        client_indices = [[] for _ in range(num_clients)]

        for client_id in range(num_clients):
            proportions = np.random.dirichlet(np.repeat(alpha, num_classes))
            
            for class_id in range(num_classes):
                class_idxs = np.where(labels == class_id)[0]
                num_samples = int(proportions[class_id] * target_size)
                
                if num_samples > 0:
                    sampled_idxs = np.random.choice(class_idxs, size=num_samples, replace=True)
                    client_indices[client_id].extend(sampled_idxs.tolist())

        return {
            client_id: np.array(client_indices[client_id], dtype='int64')
            for client_id in range(num_clients)
        }
