import torch
import torch.nn as nn

from base import Attack


class PGD(Attack):

    def __init__(
        self,
        model,
        eps = 8 / 255,
        steps = 10,
        alpha = None,
        random_start = True,
        clip_min = -100,
        clip_max = 100,
        targeted = False,
        device= None,
        transform = None,
    ) :
        """Initialize the PGD attack.

        Args:
            model: The model to attack.
            transform: A transform to normalize images.
            eps: The maximum perturbation. Defaults to 8/255.
            steps: Number of steps. Defaults to 10.
            alpha: Step size, `eps / steps` if None. Defaults to None.
            random_start: Start from random uniform perturbation. Defaults to True.
            clip_min: Minimum value for clipping. Defaults to 0.0.
            clip_max: Maximum value for clipping. Defaults to 1.0.
            targeted: Targeted attack if True. Defaults to False.
            device: Device to use for tensors. Defaults to cuda if available.
        """

        super().__init__(transform, device)

        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.targeted = targeted
        self.lossfn = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:

        # If random start enabled, delta (perturbation) is then randomly
        # initialized with samples from a uniform distribution.
        if self.random_start:
            delta = torch.empty_like(x).uniform_(-self.eps, self.eps)
            delta = torch.clamp(x + delta, self.clip_min, self.clip_max) - x
            delta.requires_grad_()
        else:
            delta = torch.zeros_like(x, requires_grad=True)

        # If alpha is not given, set to eps / steps
        if self.alpha is None:
            self.alpha = self.eps / self.steps

        # Perform PGD
        for _ in range(self.steps):
            # Compute loss
            outs = self.model(x + delta)
            loss = self.lossfn(outs, y)

            if self.targeted:
                loss = -loss

            # Compute gradient
            loss.backward()

            if delta.grad is None:
                continue

            # Update delta
            g = delta.grad.data.sign()

            delta.data = delta.data + self.alpha * g
            delta.data = torch.clamp(delta.data, -self.eps, self.eps)
            delta.data = torch.clamp(x + delta.data, self.clip_min, self.clip_max) - x

            # Zero out gradient
            delta.grad.detach_()
            delta.grad.zero_()

        return x + delta