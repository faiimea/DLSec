import torch
import torch.nn as nn

from base import Attack


class DeepFool(Attack):
    """
    The DeepFool attack.
    From the paper 'DeepFool: A Simple and Accurate Method to Fool Deep Neural Networks'
    """

    def __init__(
        self,
        model: nn.Module,
        transform:  None,
        device=None,
        steps: int = 100,
        overshoot: float = 0.02,
        num_classes: int = 10,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
    ) :

        """Initialize the DeepFool attack.

        Args:
            model: The model to attack.
            transform: A transform to normalize images.
            steps: Number of steps. Defaults to 100.
            overshoot: Overshoot parameter for noise control. Defaults to 0.02.
            num_classes: Number of classes to consider. Defaults to 10.
            clip_min: Minimum value for clipping. Defaults to 0.0.
            clip_max: Maximum value for clipping. Defaults to 1.0.
            device: Device to use for tensors. Defaults to cuda if available.
        """

        super().__init__(transform, device)

        self.model = model
        self.steps = steps
        self.overshoot = overshoot
        self.num_classes = num_classes
        self.clip_min = clip_min
        self.clip_max = clip_max

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Perform DeepFool on a batch of images.

        Args:
            x: A batch of images. Shape: (N, C, H, W).
            y: A batch of labels. Shape: (N).

        Returns:
            The perturbed images if successful. Shape: (N, C, H, W).
        """

        x.requires_grad_()
        logits = self.model(self.transform(x))

        # Get the classes
        classes = logits.argsort(axis=-1).flip(-1).detach()
        if self.num_classes is None:
            self.num_classes = logits.shape[-1]
        else:
            self.num_classes = min(self.num_classes, logits.shape[-1])

        n = len(x)
        rows = range(n)

        x0 = x
        p_total = torch.zeros_like(x)
        for _ in range(self.steps):
            # let's first get the logits using k = 1 to see if we are done
            diffs = [self._get_grads(x, 1, classes)]

            is_adv = self._is_adv(diffs[0]["logits"], y)
            if is_adv.all():
                break

            diffs += [
                self._get_grads(x, k, classes) for k in range(2, self.num_classes)
            ]

            deltas = torch.stack([d["deltas"] for d in diffs], dim=-1)
            grads = torch.stack([d["grads"] for d in diffs], dim=1)
            assert deltas.shape == (n, self.num_classes - 1)
            assert grads.shape == (n, self.num_classes - 1) + x0.shape[1:]

            # calculate the distances
            # compute f_k / ||w_k||
            distances = self._get_distances(deltas, grads)
            assert distances.shape == (n, self.num_classes - 1)

            # determine the best directions
            best = distances.argmin(1)  # compute \hat{l}
            distances = distances[rows, best]
            deltas = deltas[rows, best]
            grads = grads[rows, best]
            assert distances.shape == (n,)
            assert deltas.shape == (n,)
            assert grads.shape == x0.shape

            # apply perturbation
            distances = distances + 1e-4  # for numerical stability
            p_step = self._get_perturbations(distances, grads)  # =r_i
            assert p_step.shape == x0.shape

            p_total += p_step

            # don't do anything for those that are already adversarial
            x = torch.where(
                self._atleast_kd(is_adv, x.ndim),
                x,
                x0 + (1.0 + self.overshoot) * p_total,
            )  # =x_{i+1}

            x = (
                torch.clamp(x, self.clip_min, self.clip_max)
                .clone()
                .detach()
                .requires_grad_()
            )

        return x.detach()

    def _is_adv(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # criterion
        y_hat = logits.argmax(-1)
        is_adv = y_hat != y
        return is_adv

    def _get_deltas_logits(
        self, x: torch.Tensor, k: int, classes: torch.Tensor
    ) :
        # definition of loss_fn
        n = len(classes)
        rows = range(n)
        i0 = classes[:, 0]

        logits = self.model(self.transform(x))
        ik = classes[:, k]
        l0 = logits[rows, i0]
        lk = logits[rows, ik]
        delta_logits = lk - l0

        return {
            "sum_deltas": delta_logits.sum(),
            "deltas": delta_logits,
            "logits": logits,
        }

    def _get_grads(
    #     self, x: torch.Tensor, k: int, classes: torch.Tensor
    # ) -> dict[str, torch.Tensor]:
            self, x: torch.Tensor, k: int, classes: torch.Tensor):
        deltas_logits = self._get_deltas_logits(x, k, classes)
        deltas_logits["sum_deltas"].backward()
        if x.grad is not None:
            deltas_logits["grads"] = x.grad.clone()
            x.grad.data.zero_()
        return deltas_logits

    def _get_distances(self, deltas: torch.Tensor, grads: torch.Tensor) -> torch.Tensor:
        return abs(deltas) / (
            grads.flatten(start_dim=2, end_dim=-1).abs().sum(axis=-1)  # type: ignore
            + 1e-8
        )

    def _get_perturbations(
        self, distances: torch.Tensor, grads: torch.Tensor
    ) -> torch.Tensor:
        return self._atleast_kd(distances, grads.ndim) * grads.sign()

    def _atleast_kd(self, x: torch.Tensor, k: int) -> torch.Tensor:
        shape = x.shape + (1,) * (k - x.ndim)
        return x.reshape(shape)

