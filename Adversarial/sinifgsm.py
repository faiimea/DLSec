import torch
import torch.nn as nn

from Adversarial.base import Attack


class SINIFGSM(Attack):
    
    """
    The SI-NI-FGSM (Scale-invariant Nesterov-accelerated Iterative FGSM) attack.
    'Nesterov Accelerated Gradient and Scale Invariance for Adversarial Attacks' 
    """

    def __init__(
        self,
        model: nn.Module,
        transform=None,
        device=None,
        alpha=None,
        eps: float = 8 / 255,
        steps: int = 10,
        decay: float = 1.0,
        m: int = 3,
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
        self.m = m
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.targeted = targeted
        self.lossfn = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:


        g = torch.zeros_like(x)
        delta = torch.zeros_like(x, requires_grad=True)

        # If alpha is not given, set to eps / steps
        if self.alpha is None:
            self.alpha = self.eps / self.steps

        # Perform SI-NI-FGSM
        for _ in range(self.steps):
            # Nesterov gradient component
            nes = self.alpha * self.decay * g
            x_nes = x + delta + nes

            # Gradient is computed over scaled copies
            grad = torch.zeros_like(x)

            # Obtain scaled copies of the images
            for i in torch.arange(self.m):
                x_ness = x_nes / torch.pow(2, i)

                # Compute loss
                outs = self.model(self.transform(x_ness))
                loss = self.lossfn(outs, y)

                if self.targeted:
                    loss = -loss

                # Compute gradient
                loss.backward()

                if delta.grad is None:
                    continue

                grad += delta.grad

            # Average gradient over scaled copies
            grad /= self.m

            # Apply momentum term
            g = self.decay * grad + grad / torch.mean(
                torch.abs(grad), dim=(1, 2, 3), keepdim=True
            )

            # Update delta
            delta.data = delta.data + self.alpha * g.sign()
            delta.data = torch.clamp(delta.data, -self.eps, self.eps)
            delta.data = torch.clamp(x + delta.data, self.clip_min, self.clip_max) - x

            # Zero out gradient
            if delta.grad is not None:
                delta.grad.detach_()
                delta.grad.zero_()

        return x + delta

