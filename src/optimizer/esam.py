"""ESAM optimizer from woodenchild95/FL-Simulator (FedSpeed paper, ICLR 2023)."""

import torch


class ESAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid perturbation rate, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super().__init__(params, defaults)

        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        for group in self.param_groups:
            group["rho"] = rho
            group["adaptive"] = adaptive
        self.paras = None

    @torch.no_grad()
    def first_step(self):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-7)
            for param in group["params"]:
                if param.grad is None:
                    continue
                if group["adaptive"]:
                    e_w = torch.pow(param, 2) * param.grad * scale.to(param)
                else:
                    e_w = param.grad * scale.to(param)
                param.add_(e_w)
                self.state[param]["e_w"] = e_w

    @torch.no_grad()
    def second_step(self):
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None or not self.state[param]:
                    continue
                param.sub_(self.state[param]["e_w"])
                self.state[param]["e_w"] = 0

    def step(self, alpha=1.0):
        inputs, labels, loss_func, model = self.paras

        predictions = model(inputs)
        loss = loss_func(predictions, labels)
        self.zero_grad()
        loss.backward()

        self.first_step()

        predictions = model(inputs)
        loss = alpha * loss_func(predictions, labels)
        self.zero_grad()
        loss.backward()

        self.second_step()

    def _grad_norm(self):
        norms = []
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue
                if group["adaptive"]:
                    norms.append((torch.abs(param) * param.grad).norm(p=2))
                else:
                    norms.append(param.grad.norm(p=2))
        if not norms:
            return torch.tensor(0.0)
        return torch.norm(torch.stack(norms), p=2)
