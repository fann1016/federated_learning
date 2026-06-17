"""Flatten / unflatten model parameters (FL-Simulator vector style)."""

import torch


def params_to_vector(model, detach=True):
    if detach:
        return torch.cat([param.detach().reshape(-1) for param in model.parameters()])
    return torch.cat([param.reshape(-1) for param in model.parameters()])


def vector_to_params(model, vector):
    offset = 0
    for param in model.parameters():
        numel = param.numel()
        param.data.copy_(vector[offset : offset + numel].view_as(param))
        offset += numel


def state_dict_to_vector(state_dict):
    return torch.cat([tensor.detach().reshape(-1) for tensor in state_dict.values()])


def add_vector_to_state_dict(state_dict, vector):
    offset = 0
    updated = {}
    for key, tensor in state_dict.items():
        numel = tensor.numel()
        updated[key] = tensor + vector[offset : offset + numel].view_as(tensor)
        offset += numel
    return updated


def average_state_dicts(state_dicts):
    if not state_dicts:
        raise ValueError("average_state_dicts requires at least one state dict")
    avg = {key: value.clone().float() for key, value in state_dicts[0].items()}
    for state in state_dicts[1:]:
        for key in avg.keys():
            avg[key] += state[key].float()
    scale = 1.0 / float(len(state_dicts))
    for key in avg.keys():
        avg[key] = avg[key] * scale
    return avg
