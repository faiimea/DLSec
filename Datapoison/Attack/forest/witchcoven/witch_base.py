"""Main class, holding information about models and training/testing routines."""

import torch
import warnings

from ..utils import cw_loss
from ..consts import NON_BLOCKING, BENCHMARK
torch.backends.cudnn.benchmark = BENCHMARK

from ..victims.victim_single import _VictimSingle
from ..victims.batched_attacks import construct_attack
from ..victims.training import _split_data

class _Witch():
    """Brew poison with given arguments.

    Base class.

    This class implements _brew(), which is the main loop for iterative poisoning.
    New iterative poisoning methods overwrite the _define_objective method.

    Noniterative poison methods overwrite the _brew() method itself.

    “Double, double toil and trouble;
    Fire burn, and cauldron bubble....

    Round about the cauldron go;
    In the poison'd entrails throw.”

    """

    def __init__(self, args, setup=dict(device=torch.device('cpu'), dtype=torch.float)):
        """Initialize a model with given specs..."""
        self.args, self.setup = args, setup
        # self.retain = True if self.args.ensemble > 1 and self.args.local_rank is None else False
        self.retain = False
        self.stat_optimal_loss = None

    """ BREWING RECIPES """

    def brew(self, victim, kettle):
        """Recipe interface."""
        if len(kettle.poisonset) > 0:
            if len(kettle.targetset) > 0:
                if self.args['eps'] > 0:
                    if self.args['budget'] > 0:
                        poison_delta = self._brew(victim, kettle)
                    else:
                        poison_delta = kettle.initialize_poison(initializer='zero')
                        warnings.warn('No poison budget given. Nothing can be poisoned.')
                else:
                    poison_delta = kettle.initialize_poison(initializer='zero')
                    warnings.warn('Perturbation interval is empty. Nothing can be poisoned.')
            else:
                poison_delta = kettle.initialize_poison(initializer='zero')
                warnings.warn('Target set is empty. Nothing can be poisoned.')
        else:
            poison_delta = kettle.initialize_poison(initializer='zero')
            warnings.warn('Poison set is empty. Nothing can be poisoned.')

        return poison_delta

    def _brew(self, victim, kettle):
        """Run generalized iterative routine."""
        print(f'Starting brewing procedure ...')
        self._initialize_brew(victim, kettle)
        poisons, scores = [], torch.ones(self.args['restarts']) * 10_000

        for trial in range(self.args['restarts']):
            poison_delta, target_losses = self._run_trial(victim, kettle)
            scores[trial] = target_losses
            poisons.append(poison_delta.detach())

        optimal_score = torch.argmin(scores)
        self.stat_optimal_loss = scores[optimal_score].item()
        print(f'Poisons with minimal target loss {self.stat_optimal_loss:6.4e} selected.')
        poison_delta = poisons[optimal_score]

        return poison_delta


    def _initialize_brew(self, victim, kettle):
        """Implement common initialization operations for brewing."""
        victim.eval(dropout=True)
        # Compute target gradients
        self.targets = torch.stack([data[0] for data in kettle.targetset], dim=0).to(**self.setup)
        self.intended_classes = torch.tensor(kettle.poison_setup['intended_class']).to(device=self.setup['device'], dtype=torch.long)
        self.true_classes = torch.tensor([data[1] for data in kettle.targetset]).to(device=self.setup['device'], dtype=torch.long)

        self.target_grad, self.target_gnorm = victim.gradient(self.targets, self.intended_classes)
        print(f'Target Grad Norm is {self.target_gnorm}')

        self.target_clean_grad = None
        self.tau0 = self.args['tau'] * (self.args['poison_batch_size'] / 512)


    def _run_trial(self, victim, kettle):
        """Run a single trial."""
        poison_delta = kettle.initialize_poison().to(**self.setup)
        dataloader = kettle.poisonloader

        # poison_delta.requires_grad_()
        att_optimizer = torch.optim.Adam([poison_delta], lr=self.tau0, weight_decay=0)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(att_optimizer, milestones=[self.args['attack_iter'] // 2.667, self.args['attack_iter'] // 1.6,
                                                                                    self.args['attack_iter'] // 1.142], gamma=0.1)

        dm, ds = kettle.dm.to(device=self.setup['device']), kettle.ds.to(device=self.setup['device'])
        poison_bounds = torch.zeros_like(poison_delta)

        for step in range(self.args['attack_iter']):
            target_losses = 0
            poison_correct = 0
            poison_delta.grad = torch.zeros_like(poison_delta).to(**self.setup)
            for batch, example in enumerate(dataloader):
                loss, prediction = self._batched_step(poison_delta, poison_bounds, example, victim, kettle)
                target_losses += loss
                poison_correct += prediction

            # Note that these steps are handled batch-wise for PGD in _batched_step
            # For the momentum optimizers, we only accumulate gradients for all poisons
            # and then use optimizer.step() for the update. This is math. equivalent
            # and makes it easier to let pytorch track momentum.
            poison_delta.grad.sign_()
            att_optimizer.step()
            scheduler.step()
            att_optimizer.zero_grad()
            with torch.no_grad():
                # Projection Step
                poison_delta.data = torch.max(torch.min(poison_delta, self.args['eps'] /
                                                        ds / 255), -self.args['eps'] / ds / 255)
                poison_delta.data = torch.max(torch.min(poison_delta, (1 - dm) / ds -
                                                        poison_bounds), -dm / ds - poison_bounds)

            target_losses = target_losses / (batch + 1)
            poison_acc = poison_correct / len(dataloader.dataset)
            if step % (self.args['attack_iter'] // 5) == 0 or step == (self.args['attack_iter'] - 1):
                print(f'Iteration {step}: Target loss is {target_losses:2.4f}, '
                      f'Poison clean acc is {poison_acc * 100:2.2f}%')

        return poison_delta, target_losses



    def _batched_step(self, poison_delta, poison_bounds, example, victim, kettle):
        """Take a step toward minmizing the current target loss."""
        inputs, labels, ids = example

        inputs = inputs.to(**self.setup)
        labels = labels.to(dtype=torch.long, device=self.setup['device'], non_blocking=NON_BLOCKING)
        # Check adversarial pattern ids
        poison_slices, batch_positions = kettle.lookup_poison_indices(ids)

        # This is a no-op in single network brewing
        # In distributed brewing, this is a synchronization operation
        inputs, labels, poison_slices, batch_positions, randgen = victim.distributed_control(
            inputs, labels, poison_slices, batch_positions)

        # If a poisoned id position is found, the corresponding pattern is added here:
        if len(batch_positions) > 0:
            delta_slice = poison_delta[poison_slices].detach().to(**self.setup)
            delta_slice.requires_grad_()  # TRACKING GRADIENTS FROM HERE
            poison_images = inputs[batch_positions]
            inputs[batch_positions] += delta_slice

            # Perform differentiable data augmentation
            inputs = kettle.augment(inputs, randgen=randgen)

            # Define the loss objective and compute gradients
            loss_fn = torch.nn.CrossEntropyLoss()
            # Change loss function to include corrective terms if mixing with correction
            criterion = loss_fn

            closure = self._define_objective(inputs, labels, criterion, self.targets, self.intended_classes,
                                             self.true_classes)
            loss, prediction = victim.compute(closure, self.target_grad, self.target_clean_grad, self.target_gnorm)
            delta_slice = victim.sync_gradients(delta_slice)

            # Update Step
            poison_delta.grad[poison_slices] = delta_slice.grad.detach().to(**self.setup)
            poison_bounds[poison_slices] = poison_images.detach().to(**self.setup)
        else:
            loss, prediction = torch.tensor(0), torch.tensor(0)

        return loss.item(), prediction.item()

    def _define_objective(self):
        """Implement the closure here."""
        def closure(model, *args):
            """This function will be evaluated on all GPUs."""  # noqa: D401
            raise NotImplementedError()
            return target_loss.item(), prediction.item()

    def _pgd_step(self, delta_slice, poison_imgs, tau, dm, ds):
        """PGD step."""
        with torch.no_grad():
            # Gradient Step
            delta_slice.data -= delta_slice.grad.sign() * tau

            # Projection Step
            delta_slice.data = torch.max(torch.min(delta_slice, self.args['eps'] /
                                                   ds / 255), -self.args['eps'] / ds / 255)
            delta_slice.data = torch.max(torch.min(delta_slice, (1 - dm) / ds -
                                                   poison_imgs), -dm / ds - poison_imgs)
        return delta_slice


    def patch_targets(self, kettle):
        """Backdoor trigger attacks need to patch kettle.targets."""
        pass
