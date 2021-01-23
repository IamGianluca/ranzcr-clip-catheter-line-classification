import torch


def optimizer_factory(name, parameters):
    if name == "adam":
        return torch.optim.Adam(parameters)
    else:
        raise ValueError("Optimizer not supported yet.")
