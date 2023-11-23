import torch
import torch.nn as nn

from base import Attack


class FGSM(Attack):

    def __init__(self,model,eps=0.002,clip_min=-5,clip_max=5,targeted=False,device=None,transform=None):
        super().__init__(transform, device)
        self.model = model
        self.eps = eps
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.targeted = targeted
        self.lossfn = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:

        delta = torch.zeros_like(x, requires_grad=True)

        outs = self.model(x + delta)
        loss = self.lossfn(outs, y)

        if self.targeted:
            loss = -loss

        loss.backward()

        if delta.grad is None:
            return x

        g_sign = delta.grad.data.sign()
        delta.data = delta.data + self.eps * g_sign
        delta.data = torch.clamp(delta.data, -self.eps, self.eps)
        delta.data = torch.clamp(x + delta.data, self.clip_min, self.clip_max) - x
        return x + delta