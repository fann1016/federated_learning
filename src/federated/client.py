import copy

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from src.federated.param_utils import params_to_vector
from src.optimizer.esam import ESAM


class Client:
    def __init__(self, dataset, idxs, config):
        self.config = config
        self.device = config["device"]
        num_samples = len(idxs)
        if num_samples == 0:
            batch_size = 1
        elif config.get("batch_frac") is not None:
            batch_size = max(1, int(float(config["batch_frac"]) * num_samples))
        else:
            batch_size = int(config.get("batch_size", 32))

        self.train_loader = DataLoader(
            Subset(dataset, list(idxs)),
            batch_size=batch_size,
            shuffle=True,
        )
        self.client_control = None

    def _run_fixed_local_steps(self, model, config, step_fn):
        """Run K local SGD steps (SCAFFOLD paper: 5 steps per round)."""
        max_steps = int(config.get("local_steps", 0))
        if max_steps <= 0:
            return None

        batch_loss = []
        local_steps = 0
        data_iter = iter(self.train_loader)

        while local_steps < max_steps:
            try:
                data, target = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_loader)
                data, target = next(data_iter)

            if data.size(0) == 0:
                continue

            data, target = data.to(self.device), target.to(self.device)
            model.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)

            if torch.isnan(loss):
                print("Warning: NaN loss detected for client. Skipping batch.")
                continue

            loss.backward()
            step_fn()
            local_steps += 1
            batch_loss.append(loss.item())

        avg_loss = sum(batch_loss) / len(batch_loss) if batch_loss else 0.0
        return local_steps, avg_loss

    def _init_client_control(self, global_state):
        if self.client_control is None:
            self.client_control = {
                key: torch.zeros_like(value).to(self.device)
                for key, value in global_state.items()
            }

    def _manual_scaffold_step(self, model, global_control, client_control, lr, weight_decay):
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.grad is None:
                    continue
                grad = param.grad
                if weight_decay > 0.0:
                    grad = grad + weight_decay * param
                corrected = grad - client_control[name] + global_control[name]
                param.add_(-lr * corrected)

    def _update_client_control(self, global_state, final_state, global_control, lr, local_steps):
        denom = max(local_steps, 1) * lr
        new_control = {}
        for key in global_state.keys():
            delta_x = (global_state[key] - final_state[key]).to(self.device)
            new_control[key] = (
                self.client_control[key]
                - global_control[key]
                + delta_x / denom
            )
        return new_control

    def _train_scaffold(self, model, global_state, global_control, round_idx, config):
        lr = self._effective_lr(config, round_idx)
        weight_decay = float(config.get("weight_decay", 0.0))
        global_anchor = self._prepare_global_anchor(global_state)
        global_c = {
            key: value.to(self.device) for key, value in global_control.items()
        }
        self._init_client_control(global_state)

        num_samples = len(self.train_loader.dataset)
        if num_samples == 0:
            return model.state_dict(), 0.0, 0, {}

        def scaffold_step():
            self._manual_scaffold_step(
                model, global_c, self.client_control, lr, weight_decay
            )

        fixed = self._run_fixed_local_steps(model, config, scaffold_step)
        if fixed is not None:
            local_steps, avg_loss = fixed
        else:
            epoch_loss = []
            local_steps = 0
            for _ in range(config["local_epochs"]):
                batch_loss = []
                for data, target in self.train_loader:
                    if data.size(0) == 0:
                        continue
                    data, target = data.to(self.device), target.to(self.device)
                    model.zero_grad()
                    output = model(data)
                    loss = F.cross_entropy(output, target)

                    if torch.isnan(loss):
                        print("Warning: NaN loss detected for client. Skipping batch.")
                        continue

                    loss.backward()
                    scaffold_step()
                    local_steps += 1
                    batch_loss.append(loss.item())

                if batch_loss:
                    epoch_loss.append(sum(batch_loss) / len(batch_loss))
                else:
                    epoch_loss.append(0.0)
            avg_loss = sum(epoch_loss) / len(epoch_loss) if epoch_loss else 0.0

        final_state = model.state_dict()
        control_option = str(config.get("scaffold_control_option", "II")).upper()
        if control_option == "I":
            model.load_state_dict(global_anchor)
            full_grad = {}
            for data, target in self.train_loader:
                if data.size(0) == 0:
                    continue
                data, target = data.to(self.device), target.to(self.device)
                model.zero_grad()
                output = model(data)
                loss = F.cross_entropy(output, target)
                loss.backward()
                for name, param in model.named_parameters():
                    if param.grad is None:
                        continue
                    full_grad[name] = full_grad.get(name, 0.0) + param.grad.detach().clone()
            model.load_state_dict(final_state)
            new_control = {
                key: full_grad[key].to(self.device)
                for key in full_grad.keys()
            }
        else:
            new_control = self._update_client_control(
                global_anchor, final_state, global_c, lr, local_steps
            )
        delta_c = {
            key: (new_control[key] - self.client_control[key]).detach().cpu()
            for key in new_control.keys()
        }
        self.client_control = new_control
        return final_state, avg_loss, num_samples, delta_c

    def _effective_lr(self, config, round_idx):
        base_lr = float(config["lr"])
        lr_decay = float(config.get("lr_decay", 1.0))
        return base_lr * (lr_decay ** int(round_idx))

    def _prepare_global_anchor(self, global_state):
        if global_state is None:
            return None
        return {
            key: value.detach().to(self.device)
            for key, value in global_state.items()
        }

    def _prox_penalty(self, model, global_anchor):
        penalty = torch.tensor(0.0, device=self.device)
        for name, param in model.named_parameters():
            penalty = penalty + torch.sum((param - global_anchor[name]) ** 2)
        return penalty

    def _manual_fedcm_step(self, model, server_delta, alpha, lr, weight_decay):
        """FedCM local step: x <- x - eta_l * (alpha * g_i(x) + (1-alpha) * Delta^t)."""
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.grad is None:
                    continue
                grad = param.grad
                if weight_decay > 0.0:
                    grad = grad + weight_decay * param
                direction = server_delta[name]
                blended = alpha * grad + (1.0 - alpha) * direction
                param.add_(-lr * blended)

    def _train_fedcm(self, model, server_delta, round_idx, config):
        alpha = float(config.get("fedcm_alpha", 0.1))
        lr = self._effective_lr(config, round_idx)
        weight_decay = float(config.get("weight_decay", 0.0))

        epoch_loss = []
        num_samples = len(self.train_loader.dataset)
        local_steps = 0
        if num_samples == 0:
            return model.state_dict(), 0.0, 0, 0

        for _ in range(config["local_epochs"]):
            batch_loss = []
            for data, target in self.train_loader:
                if data.size(0) == 0:
                    continue
                data, target = data.to(self.device), target.to(self.device)
                model.zero_grad()
                output = model(data)
                loss = F.cross_entropy(output, target)

                if torch.isnan(loss):
                    print("Warning: NaN loss detected for client. Skipping batch.")
                    continue

                loss.backward()
                self._manual_fedcm_step(model, server_delta, alpha, lr, weight_decay)
                local_steps += 1
                batch_loss.append(loss.item())

            if batch_loss:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
            else:
                epoch_loss.append(0.0)

        avg_loss = sum(epoch_loss) / len(epoch_loss) if epoch_loss else 0.0
        return model.state_dict(), avg_loss, num_samples, local_steps

    def _train_fedspeed(self, model, global_state, dual_correction, round_idx, config):
        """
        FedSpeed local training (FL-Simulator client/fedspeed.py):
        ESAM perturbation + prox-correction via lamb * <w, Local_dual_correction>.
        """
        lr = self._effective_lr(config, round_idx)
        weight_decay = float(config.get("weight_decay", 0.0))
        lamb = float(config.get("lamb", config.get("mu", 0.1)))
        sam_rho = float(config.get("sam_rho", 0.1))
        sam_alpha = float(config.get("sam_alpha", 1.0))
        grad_clip = float(config.get("grad_clip_norm", 10.0))

        global_vec = params_to_vector(model).detach()
        dual_correction = dual_correction.to(self.device)

        base_optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay + lamb,
        )
        esam_optimizer = ESAM(model.parameters(), base_optimizer, rho=sam_rho)
        loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")

        epoch_loss = []
        num_samples = len(self.train_loader.dataset)
        if num_samples == 0:
            return model.state_dict(), 0.0, 0, torch.zeros_like(global_vec)

        for _ in range(config["local_epochs"]):
            batch_loss = []
            for data, target in self.train_loader:
                if data.size(0) == 0:
                    continue
                data, target = data.to(self.device), target.to(self.device)

                esam_optimizer.paras = [data, target, loss_fn, model]
                esam_optimizer.step(alpha=sam_alpha)

                param_vec = params_to_vector(model, detach=False)
                loss_correct = lamb * torch.sum(param_vec * dual_correction)
                loss_correct.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                base_optimizer.step()
                batch_loss.append(loss_correct.item())

            if batch_loss:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
            else:
                epoch_loss.append(0.0)

        final_vec = params_to_vector(model)
        local_update = final_vec - global_vec
        avg_loss = sum(epoch_loss) / len(epoch_loss) if epoch_loss else 0.0
        return model.state_dict(), avg_loss, num_samples, local_update.detach().cpu()

    def train(
        self,
        model,
        global_state=None,
        server_delta=None,
        global_control=None,
        dual_correction=None,
        round_idx=0,
        client_config=None,
    ):
        model.to(self.device)
        model.train()

        config = client_config if client_config is not None else self.config
        algorithm = str(config.get("algorithm", "FedAvg")).lower()

        if algorithm == "fedcm":
            if server_delta is None:
                raise ValueError("FedCM requires server_delta (Δ^t) from the server.")
            return self._train_fedcm(model, server_delta, round_idx, config)

        if algorithm == "scaffold":
            if global_state is None or global_control is None:
                raise ValueError("SCAFFOLD requires global_state and global_control from the server.")
            return self._train_scaffold(
                model, global_state, global_control, round_idx, config
            )

        if algorithm == "fedspeed":
            if global_state is None or dual_correction is None:
                raise ValueError("FedSpeed requires global_state and dual_correction from the server.")
            return self._train_fedspeed(
                model, global_state, dual_correction, round_idx, config
            )

        mu = float(config.get("mu", 0.0))
        use_prox = algorithm == "fedprox" and mu > 0.0
        global_anchor = self._prepare_global_anchor(global_state) if use_prox else None

        effective_lr = self._effective_lr(config, round_idx)
        momentum = float(config.get("optimizer_momentum", 0.0))
        weight_decay = float(config.get("weight_decay", 0.0))
        nesterov = config.get("client_nesterov", False)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=effective_lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov if momentum > 0 else False,
        )

        epoch_loss = []
        num_samples = len(self.train_loader.dataset)
        if num_samples == 0:
            return model.state_dict(), 0.0, 0

        for _ in range(config["local_epochs"]):
            batch_loss = []
            for data, target in self.train_loader:
                if data.size(0) == 0:
                    continue
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = model(data)
                loss = F.cross_entropy(output, target)

                if use_prox:
                    loss = loss + (mu / 2.0) * self._prox_penalty(model, global_anchor)

                if torch.isnan(loss):
                    print("Warning: NaN loss detected for client. Skipping batch.")
                    continue

                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())

            if batch_loss:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
            else:
                epoch_loss.append(0.0)

        avg_loss = sum(epoch_loss) / len(epoch_loss) if epoch_loss else 0.0
        return model.state_dict(), avg_loss, num_samples
