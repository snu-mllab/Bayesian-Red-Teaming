from gpytorch.mlls import ExactMarginalLogLikelihood
import torch

print_freq = 20

from typing import List

def fit_model(
    surrogate_model = None,
    train_X : torch.Tensor = None,
    fit_iter : int = 20,
    ):
    # Use the adam optimizer
    optimizer = torch.optim.Adam([
    {'params': surrogate_model.parameters()},  # Includes GaussianLikelihood parameters
    ], lr=0.1)

    mll = ExactMarginalLogLikelihood(surrogate_model.likelihood, surrogate_model).cuda()
    for i in range(fit_iter):
        optimizer.zero_grad()
        output = surrogate_model(train_X)
        loss = -mll(output, surrogate_model.train_targets)
        loss.backward()
        optimizer.step()
    return surrogate_model