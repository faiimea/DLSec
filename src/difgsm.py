import torch
import torch.nn as nn
import torch.nn.functional as f

from base import Attack


class DIFGSM(Attack):

    def __init__(
        self,
        model: nn.Module,
        transform = None,
        alpha = None,
        device=None,
        eps: float = 8 / 255,
        steps: int = 10,
        decay: float = 1.0,
        resize_rate: float = 0.9,
        diversity_prob: float = 1.0,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
        targeted: bool = False,
    ) :
        super().__init__(transform, device)

        self.model = model
        self.eps = eps
        self.steps = steps
        self.alpha = alpha
        self.decay = decay
        self.resize_rate = resize_rate
        self.diversity_prob = diversity_prob
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

        # Perform DI-FGSM
        for _ in range(self.steps):
            # Apply input diversity to intermediate images
            x_adv = input_diversity(x + delta, self.resize_rate, self.diversity_prob)

            # Compute loss
            outs = self.model(self.transform(x_adv))
            loss = self.lossfn(outs, y)

            if self.targeted:
                loss = -loss

            # Compute gradient
            loss.backward()

            if delta.grad is None:
                continue

            # Apply momentum term
            g = self.decay * g + delta.grad / torch.mean(
                torch.abs(delta.grad), dim=(1, 2, 3), keepdim=True
            )

            # Update delta
            delta.data = delta.data + self.alpha * g.sign()
            delta.data = torch.clamp(delta.data, -self.eps, self.eps)
            delta.data = torch.clamp(x + delta.data, self.clip_min, self.clip_max) - x

            # Zero out gradient
            delta.grad.detach_()
            delta.grad.zero_()

        return x + delta


def input_diversity(
    x: torch.Tensor, resize_rate: float = 0.9, diversity_prob: float = 0.5
) -> torch.Tensor:

    if torch.rand(1) < diversity_prob:
        return x

    img_size = x.shape[-1]
    img_resize = int(img_size * resize_rate)

    if resize_rate < 1:
        img_size = img_resize
        img_resize = x.shape[-1]

    rnd = torch.randint(low=img_size, high=img_resize, size=(1,), dtype=torch.int32)
    rescaled = f.interpolate(x, size=[rnd, rnd], mode="nearest")

    h_rem = img_resize - rnd
    w_rem = img_resize - rnd

    pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
    pad_bottom = h_rem - pad_top
    pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
    pad_right = w_rem - pad_left

    pad = [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()]
    padded = f.pad(rescaled, pad=pad, mode="constant", value=0)

    return padded



