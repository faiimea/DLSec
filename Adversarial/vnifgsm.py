import torch
import torch.nn as nn

from Adversarial.base import Attack


class VNIFGSM(Attack):

    def __init__(
        self,
        model: nn.Module,
        transform=None,
        device=None,
        alpha=None,
        eps: float = 8 / 255,
        steps: int = 10,
        decay: float = 1.0,
        n: int = 5,
        beta: float = 1.5,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
        targeted: bool = False,
    ) -> None:


        super().__init__(transform, device)

        self.model = model
        self.eps = eps
        self.steps = steps
        self.alpha = alpha
        self.decay = decay
        self.n = n
        self.beta = beta
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.targeted = targeted
        self.lossfn = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:


        g = torch.zeros_like(x)  # Momentum
        v = torch.zeros_like(x)  # Gradient variance
        delta = torch.zeros_like(x, requires_grad=True)

        # If alpha is not given, set to eps / steps
        if self.alpha is None:
            self.alpha = self.eps / self.steps

        # Perform VNI-FGSM
        for _ in range(self.steps):
            # Nesterov gradient component
            nes = self.alpha * self.decay * g
            x_nes = x + delta + nes

            # Compute loss
            outs = self.model(self.transform(x_nes))
            loss = self.lossfn(outs, y)

            if self.targeted:
                loss = -loss

            # Compute gradient
            loss.backward()

            if delta.grad is None:
                continue

            # Apply momentum term and variance
            delta_grad = delta.grad
            g = self.decay * g + (delta_grad + v) / torch.mean(
                torch.abs(delta_grad + v), dim=(1, 2, 3), keepdim=True
            )

            # Compute gradient variance
            gv_grad = torch.zeros_like(x)
            for _ in range(self.n):
                # Get neighboring samples perturbation
                neighbors = delta.data + torch.randn_like(x).uniform_(
                    -self.eps * self.beta, self.eps * self.beta
                )
                neighbors.requires_grad_()
                neighbor_outs = self.model(self.transform(x + neighbors))
                neighbor_loss = self.lossfn(neighbor_outs, y)

                if self.targeted:
                    neighbor_loss = -neighbor_loss

                neighbor_loss.backward()

                gv_grad += neighbors.grad

            # Accumulate gradient variance into v
            v = gv_grad / self.n - delta_grad

            # Update delta
            delta.data = delta.data + self.alpha * g.sign()
            delta.data = torch.clamp(delta.data, -self.eps, self.eps)
            delta.data = torch.clamp(x + delta.data, self.clip_min, self.clip_max) - x

            # Zero out gradient
            delta.grad.detach_()
            delta.grad.zero_()

        return x + delta

