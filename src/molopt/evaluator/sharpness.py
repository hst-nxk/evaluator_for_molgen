"""IGNITE: penalise the surrogate's output gradient norm, to make the model less "sharp".

Source: Dao MC, Nguyen PL, Truong TN, Hoang TN. *Incorporating Surrogate Gradient Norm to
Improve Offline Optimization Techniques*. NeurIPS 2024, DOI 10.52202/079017-0258,
arXiv:2503.04242.

The paper's idea: a surrogate trained on offline data is usually untrustworthy in regions
outside the data it has seen. The paper measures how "sharp" a surrogate is with

    R_D(omega) = max_{||delta|| <= rho} | E_x[g(x; omega + delta)] - E_x[g(x; omega)] |

and approximates it by ``rho * ||grad_omega h(omega)||`` with
``h(omega) = E_x[g(x; omega)]`` (equations 10, 18). So instead of minimising the loss alone,
the paper solves the constrained problem ``rho * ||grad_omega h(omega)|| <= epsilon``
(equation 11) with a Lagrange multiplier ``lambda`` updated by dual ascent (equations
12-14).

The key implementation point: the gradient of the penalty term itself would need a Hessian,
so the paper approximates it by a finite difference around the perturbed point
``omega + r * g2 / ||g2||`` (equations 83-85, Algorithm 1). Every step then needs
first-order gradients only, at the cost of one extra forward pass.

The total gradient of one step, exactly as in the paper's Algorithm 1:

    g1 = grad_omega (the mean loss over the batch)
    g2 = grad_omega (the mean output over the batch)      # = grad_omega h
    omega_hat = omega + r * g2 / ||g2||
    g3 = grad_omega (the mean output over the batch, evaluated at omega_hat)
    g  = g1 + lambda * (rho / r) * (g3 - g2)
    lambda <- lambda + eta_lambda * (rho * ||g2|| - epsilon)

Two deliberate departures from the paper, recorded in the config's
``stage3.train.sharpness.note``: the paper updates the parameters by plain gradient descent
(this uses AdamW, to match every earlier training run), and the paper's assumption 1 needs
the output bounded in [0, 1] while this project's Δp labels are unbounded - so the theory
does not apply directly and the effect has to be measured by ablation.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

SETTING_ENABLED = "enabled"


def is_enabled(settings: Mapping[str, Any], override: str | None = None) -> bool:
    """Whether the penalty term is on: ``override`` ('on'/'off') beats the config."""
    if override == "on":
        return True
    if override == "off":
        return False
    return bool(settings[SETTING_ENABLED])


class SharpnessPenalty:
    """One IGNITE step: compute the total gradient and write it into each parameter's
    ``.grad``.

    Takes: the config's ``stage3.train.sharpness`` block.
    Holds state: ``lambda_value``, the Lagrange multiplier, updated after every batch.
    """

    def __init__(self, settings: Mapping[str, Any]) -> None:
        self.rho = float(settings["rho"])
        self.radius = float(settings["finite_difference_radius"])
        self.epsilon = float(settings["epsilon"])
        self.lambda_learning_rate = float(settings["lambda_learning_rate"])
        self.lambda_value = float(settings["lambda_init"])
        if self.radius <= 0.0:
            raise ValueError("finite_difference_radius has to be > 0: it is the "
                             "denominator of the finite difference")
        if self.rho <= 0.0:
            raise ValueError("rho has to be > 0")
        self.last_gradient_norm = 0.0

    def apply(self, loss: Any, prediction: Any, parameters: Sequence[Any],
              recompute_mean: Callable[[], Any], share: float = 1.0,
              accumulate: bool = False) -> None:
        """Write the total gradient into ``.grad``, then update ``lambda``.

        Takes: the batch's loss; the batch's outputs (not yet averaged); the list of
        parameters that need gradients; a callable that runs the forward pass again on THAT
        SAME batch and returns the mean output (needed because the finite difference has to
        be measured at the perturbed parameters).

        ``share`` multiplies the total gradient, and ``accumulate`` ADDS to the existing
        ``.grad`` instead of overwriting it. The two exist for accumulating gradients across
        several sub-batches: overwriting would erase the earlier sub-batches' gradients with
        no warning. The weight has to multiply the TOTAL gradient and not the ``loss``,
        because the penalty term is computed from ``prediction.mean()`` and is therefore
        unaffected by splitting the loss up.
        """
        import torch

        trainable = [parameter for parameter in parameters if parameter.requires_grad]
        if not trainable:
            raise ValueError("no parameter needs a gradient")
        loss_grad = torch.autograd.grad(loss, trainable, retain_graph=True)
        output_grad = torch.autograd.grad(prediction.mean(), trainable)
        norm = torch.sqrt(sum((part * part).sum() for part in output_grad))
        self.last_gradient_norm = float(norm.detach().to("cpu"))

        if self.last_gradient_norm > 0.0:
            scale = self.radius / self.last_gradient_norm
            with torch.no_grad():
                for parameter, part in zip(trainable, output_grad):
                    parameter.add_(part, alpha=scale)
            try:
                shifted_grad = torch.autograd.grad(recompute_mean(), trainable)
            finally:
                # The parameters have to be restored even if the extra forward pass
                # raises, or the model would be left standing at the perturbed point.
                with torch.no_grad():
                    for parameter, part in zip(trainable, output_grad):
                        parameter.sub_(part, alpha=scale)
            weight = self.lambda_value * self.rho / self.radius
            total = [first + weight * (third - second)
                     for first, second, third in zip(loss_grad, output_grad, shifted_grad)]
        else:
            # ||g2|| = 0: the penalty term has no direction to move in, so only the loss
            # gradient is left.
            total = list(loss_grad)

        for parameter, part in zip(trainable, total):
            scaled = part if share == 1.0 else part * share
            if accumulate and parameter.grad is not None:
                parameter.grad = parameter.grad + scaled
            else:
                parameter.grad = scaled

        # Dual ascent. The paper does not say whether lambda is bounded below; it is
        # clamped at 0 here because the Lagrangian of equation 12 requires lambda > 0.
        violation = self.rho * self.last_gradient_norm - self.epsilon
        self.lambda_value = max(0.0, self.lambda_value
                                + self.lambda_learning_rate * violation)

    def state(self) -> dict[str, float]:
        """The state to record in the run's manifest."""
        return {"lambda": self.lambda_value,
                "output_gradient_norm": self.last_gradient_norm,
                "sharpness_estimate": self.rho * self.last_gradient_norm}
