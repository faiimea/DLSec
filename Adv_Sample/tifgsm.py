import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f

from base import Attack


class TIFGSM(Attack):

    def __init__(
        self,
        model: nn.Module,
        transform=None,
        device=None,
        alpha=None,
        eps: float = 8 / 255,
        steps: int = 10,
        decay: float = 1.0,
        kern_len: int = 15,
        n_sig: int = 3,
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
        self.kern_len = kern_len
        self.n_sig = n_sig
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.targeted = targeted
        self.lossfn = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:

        g = torch.zeros_like(x)
        delta = torch.zeros_like(x, requires_grad=True)

        # Get kernel
        kernel = self.get_kernel()

        # If alpha is not given, set to eps / steps
        if self.alpha is None:
            self.alpha = self.eps / self.steps

        # Perform TI-FGSM
        for _ in range(self.steps):
            # Compute loss
            outs = self.model(self.transform(x + delta))
            loss = self.lossfn(outs, y)

            if self.targeted:
                loss = -loss

            # Compute gradient
            loss.backward()

            if delta.grad is None:
                continue

            # Apply kernel to gradient
            g = f.conv2d(delta.grad, kernel, stride=1, padding="same", groups=3)

            # Apply momentum term
            g = self.decay * g + g / torch.mean(
                torch.abs(g), dim=(1, 2, 3), keepdim=True
            )

            # Update delta
            delta.data = delta.data + self.alpha * g.sign()
            delta.data = torch.clamp(delta.data, -self.eps, self.eps)
            delta.data = torch.clamp(x + delta.data, self.clip_min, self.clip_max) - x

            # Zero out gradient
            delta.grad.detach_()
            delta.grad.zero_()

        return x + delta

    def get_kernel(self) -> torch.Tensor:
        kernel = self.gkern(self.kern_len, self.n_sig).astype(np.float32)

        kernel = np.expand_dims(kernel, axis=0)  # (W, H) -> (1, W, H)
        kernel = np.repeat(kernel, 3, axis=0)  # -> (C, W, H)
        kernel = np.expand_dims(kernel, axis=1)  # -> (C, 1, W, H)
        return torch.from_numpy(kernel).to(self.device)

    @staticmethod
    def gkern(kern_len: int = 15, n_sig: int = 3) -> np.ndarray:
        """Return a 2D Gaussian kernel array."""

        import scipy.stats as st

        interval = (2 * n_sig + 1.0) / kern_len
        x = np.linspace(-n_sig - interval / 2.0, n_sig + interval / 2.0, kern_len + 1)
        kern1d = np.diff(st.norm.cdf(x))
        kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
        kernel = kernel_raw / kernel_raw.sum()
        return kernel

