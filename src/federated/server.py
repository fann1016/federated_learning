import copy

import torch
import torch.nn.functional as F

from src.federated.param_utils import (
    add_vector_to_state_dict,
    average_state_dicts,
    state_dict_to_vector,
)


class Server:
    def __init__(self, model, config):
        self.config = config
        self.device = config["device"]
        self.global_model = model.to(self.device)
        self.beta = float(config.get("server_momentum_beta", 0.0))
        self.server_lr = float(config.get("server_lr", 1.0))
        self.server_nesterov = bool(config.get("server_nesterov", False))
        self.v = None
        self.delta = None
        self.global_control = None
        self.lr_decay = float(config.get("lr_decay", 1.0))
        self.h_params_list = None
        if str(config.get("algorithm", "")).lower() == "fedspeed":
            self._init_fedspeed_storage()

    def _init_fedspeed_storage(self):
        num_clients = int(self.config["num_clients"])
        param_dim = sum(param.numel() for param in self.global_model.parameters())
        self.h_params_list = torch.zeros(
            (num_clients, param_dim), dtype=torch.float32, device=self.device
        )

    def get_fedspeed_dual_correction(self, client_idx, global_vec):
        """Local_dual_correction = h_i - x^t (FL-Simulator FedSpeed.process_for_communication)."""
        if self.h_params_list is None:
            raise RuntimeError("FedSpeed dual correction requested before server init.")
        return self.h_params_list[int(client_idx)] - global_vec.to(self.device)

    def update_fedspeed_h(self, client_idx, local_update_vec):
        self.h_params_list[int(client_idx)] += local_update_vec.to(self.device)

    def aggregate_fedspeed(self, local_weights):
        """
        FedSpeed global update (FL-Simulator server/FedSpeed.global_update):
        w^{t+1} = mean_i(w_i^K) + mean_j(h_j)
        """
        if self.h_params_list is None:
            raise RuntimeError("FedSpeed aggregation requested before server init.")

        avg_weights = average_state_dicts(local_weights)
        h_mean = self.h_params_list.mean(dim=0)
        new_state = add_vector_to_state_dict(avg_weights, h_mean)
        self.global_model.load_state_dict(new_state)

    def global_param_vector(self):
        return state_dict_to_vector(self.global_model.state_dict()).to(self.device)

    def _init_delta(self):
        if self.delta is None:
            self.delta = {
                key: torch.zeros_like(value).to(self.device)
                for key, value in self.global_model.state_dict().items()
            }

    def _init_global_control(self):
        if self.global_control is None:
            self.global_control = {
                key: torch.zeros_like(value).to(self.device)
                for key, value in self.global_model.state_dict().items()
            }

    def get_server_delta(self):
        """Return Δ^t for FedCM client updates."""
        self._init_delta()
        return {
            key: value.detach().clone()
            for key, value in self.delta.items()
        }

    def get_global_control(self):
        """Return server control variate c for SCAFFOLD."""
        self._init_global_control()
        return {
            key: value.detach().clone()
            for key, value in self.global_control.items()
        }

    def _round_lr(self, round_idx):
        return float(self.config["lr"]) * (self.lr_decay ** int(round_idx))

    def aggregate_fedcm(self, local_weights, local_sizes, local_steps, global_state_before, round_idx=0):
        """
        FedCM server update (Algorithm 2, arXiv:2106.10874).

        Let p_i = n_i / sum_j n_j and K_eff = sum_i p_i K_i.
        The server pseudo-gradient is
            Delta^{t+1} = sum_i p_i (x^t - x_i^{t,K_i}) / (eta_l^t K_eff)
        and the global model is updated as FedAvg:
            x^{t+1} = x^t - eta_g eta_l^t K_eff Delta^{t+1}
                    = x^t + eta_g sum_i p_i (x_i^{t,K_i} - x^t).
        """
        self._init_delta()
        total_size = sum(local_sizes)
        if total_size <= 0:
            return

        avg_delta_w = copy.deepcopy(local_weights[0])
        for key in avg_delta_w.keys():
            avg_delta_w[key] = (
                local_weights[0][key] - global_state_before[key]
            ) * (local_sizes[0] / total_size)
            for i in range(1, len(local_weights)):
                avg_delta_w[key] += (
                    local_weights[i][key] - global_state_before[key]
                ) * (local_sizes[i] / total_size)

        weighted_steps = sum(
            size * steps for size, steps in zip(local_sizes, local_steps)
        )
        if weighted_steps <= 0:
            return

        lr_scale = self._round_lr(round_idx)
        k_eff = weighted_steps / total_size
        new_delta = {}
        for key in avg_delta_w.keys():
            new_delta[key] = (
                -avg_delta_w[key].to(self.device) / (lr_scale * k_eff)
            )

        self.delta = new_delta

        current = {
            key: value.to(self.device)
            for key, value in global_state_before.items()
        }
        new_state = {}
        for key in current.keys():
            new_state[key] = current[key] + self.server_lr * avg_delta_w[key].to(self.device)

        self.global_model.load_state_dict(new_state)

    def aggregate_scaffold(self, local_weights, local_sizes, control_deltas, num_clients):
        """
        SCAFFOLD server update (Karimireddy et al., ICML 2020, Eq. 5).

        c <- c + (1/N) sum(c_i^+ - c_i)
        """
        total_size = sum(local_sizes)
        if total_size <= 0:
            return

        avg_weights = copy.deepcopy(local_weights[0])
        for key in avg_weights.keys():
            avg_weights[key] = avg_weights[key] * (local_sizes[0] / total_size)
            for i in range(1, len(local_weights)):
                avg_weights[key] += local_weights[i][key] * (local_sizes[i] / total_size)

        self.global_model.load_state_dict(avg_weights)

        if not control_deltas:
            return

        self._init_global_control()
        scale = 1.0 / float(num_clients)
        for key in self.global_control.keys():
            update = torch.zeros_like(self.global_control[key])
            for delta_c in control_deltas:
                update += delta_c[key].to(self.device)
            self.global_control[key] += scale * update

    def aggregate(self, local_weights, local_sizes):
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
            delta = (current[k] - avg_weights[k]).to(self.device)
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
                test_loss += F.cross_entropy(output, target, reduction="sum").item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        accuracy = 100.0 * correct / len(test_loader.dataset)
        return test_loss, accuracy
