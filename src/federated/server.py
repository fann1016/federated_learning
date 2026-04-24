import torch
import torch.nn.functional as F
import copy

class Server:
    def __init__(self, model, config):
        self.config = config
        self.device = config["device"]
        self.global_model = model.to(self.device)
        self.beta = float(config.get("server_momentum_beta", 0.0))
        self.server_lr = float(config.get("server_lr", 1.0))
        self.server_nesterov = bool(config.get("server_nesterov", False))
        self.v = None

    def aggregate(self, local_weights, local_sizes):
        # Weighted aggregation: w = sum( (n_k / n) * w_k )
        total_size = sum(local_sizes)
        avg_weights = copy.deepcopy(local_weights[0])
        
        for key in avg_weights.keys():
            avg_weights[key] = avg_weights[key] * (local_sizes[0] / total_size)
            for i in range(1, len(local_weights)):
                avg_weights[key] += local_weights[i][key] * (local_sizes[i] / total_size)

        if self.beta <= 0.0:
            self.global_model.load_state_dict(avg_weights)
            return

        current = self.global_model.state_dict()
        if self.v is None:
            self.v = {k: torch.zeros_like(v).to(self.device) for k, v in current.items()}

        new_state = {}
        for k in current.keys():
            # Delta_w = w_global - w_avg (as per paper Section 4.2)
            delta = (current[k] - avg_weights[k]).to(self.device)
            # Momentum buffer on the aggregated server update.
            self.v[k] = self.beta * self.v[k] + delta
            if self.server_nesterov:
                update = delta + self.beta * self.v[k]
            else:
                update = self.v[k]
            new_state[k] = current[k] - self.server_lr * update

        self.global_model.load_state_dict(new_state)

    def evaluate(self, test_loader):
        self.global_model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.global_model(data)
                test_loss += F.cross_entropy(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)
        return test_loss, accuracy
