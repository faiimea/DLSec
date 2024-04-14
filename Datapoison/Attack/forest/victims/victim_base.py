"""Base victim class."""

import torch

from .models import get_model
from .training import get_optimizers, run_step
from ..hyperparameters import training_strategy
from ..utils import average_dicts
from ..consts import BENCHMARK, SHARING_STRATEGY
torch.backends.cudnn.benchmark = BENCHMARK
torch.multiprocessing.set_sharing_strategy(SHARING_STRATEGY)


FINETUNING_LR_DROP = 0.001


class _VictimBase:
    """Implement model-specific code and behavior.

    Expose:
    Attributes:
     - model
     - optimizer
     - scheduler
     - criterion

     Methods:
     - initialize
     - train
     - retrain
     - validate
     - iterate

     - compute
     - gradient
     - eval

     Internal methods that should ideally be reused by other backends:
     - _initialize_model
     - _step
    """
    def __init__(self, args, setup=dict(device=torch.device('cpu'), dtype=torch.float)):
        """Initialize empty victim."""
        self.args, self.setup = args, setup
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.initialize(pretrain=self.args['pretrained'])

    def gradient(self, images, labels):
        """Compute the gradient of criterion(model) w.r.t to given data."""
        raise NotImplementedError()
        return grad, grad_norm

    def compute(self, function):
        """Compute function on all models.

        Function has arguments: model, ...
        """
        raise NotImplementedError()

    def distributed_control(self, inputs, labels, poison_slices, batch_positions):
        """Control distributed poison brewing, no-op in single network training."""
        randgen = None
        return inputs, labels, poison_slices, batch_positions, randgen

    def sync_gradients(self, input):
        """Sync gradients of given variable. No-op for single network training."""
        return input

    def reset_learning_rate(self):
        """Reset scheduler object to initial state."""
        raise NotImplementedError()

    """ Methods to initialize and modify a model."""

    def initialize(self, seed=None):
        raise NotImplementedError()

    def reinitialize_last_layer(self, seed=None):
        raise NotImplementedError()

    def freeze_feature_extractor(self):
        raise NotImplementedError()

    def save_feature_representation(self):
        raise NotImplementedError()

    def load_feature_representation(self):
        raise NotImplementedError()


    """ METHODS FOR (CLEAN) TRAINING AND TESTING OF BREWED POISONS"""

    def train(self, kettle, max_epoch=None):
        """Clean (pre)-training of the chosen model, no poisoning involved."""
        print('Starting clean training ...')
        stats_clean = self._iterate(kettle, poison_delta=None, max_epoch=max_epoch,
                                    pretraining_phase=False)

        if self.args['scenario'] == 'transfer':
            self.save_feature_representation()
            self.freeze_feature_extractor()
            self.eval()
            print('Features frozen.')

        return stats_clean

    def retrain(self, kettle, poison_delta):
        """Check poison on the initialization it was brewed on."""
        if self.args['scenario'] == 'from-scratch':
            self.initialize(seed=self.model_init_seed)
            print('Model re-initialized to initial seed.')
        elif self.args['scenario'] == 'transfer':
            self.load_feature_representation()
            self.reinitialize_last_layer(reduce_lr_factor=1.0, seed=self.model_init_seed)
            print('Linear layer reinitialized to initial seed.')
        return self._iterate(kettle, poison_delta=poison_delta)

    def validate(self, kettle, poison_delta):
        """Check poison on a new initialization(s), depending on the scenario."""
        run_stats = list()

        for runs in range(self.args['vruns']):
            if self.args['scenario'] == 'from-scratch':
                self.initialize()
                print('Model reinitialized to random seed.')
            elif self.args['scenario'] == 'transfer':
                self.load_feature_representation()
                self.reinitialize_last_layer(reduce_lr_factor=1.0)
                print('Linear layer reinitialized to initial seed.')

            # Train new model
            run_stats.append(self._iterate(kettle, poison_delta=poison_delta, max_epoch=self.args['epochs']))
            print(run_stats)
        return average_dicts(run_stats)

    def eval(self, dropout=True):
        """Switch everything into evaluation mode."""
        raise NotImplementedError()

    def _iterate(self, kettle, poison_delta):
        """Validate a given poison by training the model and checking target accuracy."""
        raise NotImplementedError()

    def _adversarial_step(self, kettle, poison_delta, step, poison_targets, true_classes):
        """Step through a model epoch to in turn minimize target loss."""
        raise NotImplementedError()

    def _initialize_model(self, local_model, optimizer, lr, weight_decay, epochs, frozen):
        model = local_model     # 本来是从models.py里，get_model()函数返回的model，作者又封装了一遍……麻了
        model.frozen = frozen    # 貌似只是个标记
        # Define training routine
        # defs = training_strategy(model_name, self.args)
        optimizer, scheduler = get_optimizers(model, optimizer, lr, weight_decay, epochs)

        # return model, defs, optimizer, scheduler
        return model, optimizer, scheduler


    def _step(self, kettle, poison_delta, epoch, stats, model, optimizer, scheduler, pretraining_phase=False):
        """Single epoch. Can't say I'm a fan of this interface, but ..."""
        run_step(kettle, poison_delta, epoch, stats, model, optimizer, scheduler,
                 loss_fn=self.loss_fn, pretraining_phase=pretraining_phase)
