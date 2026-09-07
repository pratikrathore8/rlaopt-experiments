"""Run the paper's lasso tuning example and save its objective values for plotting."""

import json
from pathlib import Path

import torch
from rlaopt.atoms import L1Norm, SumSquares
from rlaopt.expression import Variable
from rlaopt.solvers import ProxGrad, ProxGradConfig


def main():
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    rng = torch.Generator().manual_seed(0)
    n_train, n_val, p, support = 512, 128, 64, 16
    beta = torch.zeros(p)
    beta[torch.randperm(p, generator=rng)[:support]] = torch.randn(support, generator=rng) / support**0.5
    X = torch.randn(n_train + n_val, p, generator=rng)
    y = X @ beta + 0.1 * torch.randn(n_train + n_val, generator=rng)
    X_train, X_val = X[:n_train], X[n_train:]
    y_train, y_val = y[:n_train], y[n_train:]
    config = ProxGradConfig(
        eta=float(n_train / (2 * torch.linalg.matrix_norm(X_train, ord=2)**2)),
        use_linesearch=False, use_acceleration=False,
    )

    def outer_objective(mu):
        w = Variable(torch.zeros(p), name="w")
        objective = SumSquares(X_train @ w - y_train) * (1 / n_train) + L1Norm(w, scaling=mu)
        solver = ProxGrad(objective, config, detach=False)
        values = objective.variable_values
        state = solver.init_state(values)
        for _ in range(100):
            values, state = solver.step(values, state)
        return ((X_val @ values["w"] - y_val)**2).mean()

    mu = torch.tensor(0.2)
    grad_and_value = torch.func.grad_and_value(outer_objective)
    trace = []
    for iteration in range(41):
        grad, value = grad_and_value(mu)
        trace.append({"iteration": iteration, "mu": float(mu), "validation_mse": float(value)})
        if iteration < 40:
            mu = (mu - 0.05 * grad).detach()

    output = Path("artifacts/differentiable-optimization/result.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"trace": trace}, indent=2) + "\n")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
