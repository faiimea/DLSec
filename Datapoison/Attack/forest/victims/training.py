"""Repeatable code parts concerning optimization and training schedules."""

import torch

from collections import defaultdict

from .utils import print_and_save_stats
from .batched_attacks import construct_attack
from tqdm import tqdm

from ..consts import NON_BLOCKING, BENCHMARK
torch.backends.cudnn.benchmark = BENCHMARK

def run_step(kettle, poison_delta, epoch, stats, model, optimizer, scheduler, loss_fn, pretraining_phase=False):

    epoch_loss, total_preds, correct_preds = 0, 0, 0

    if pretraining_phase:
        train_loader = kettle.pretrainloader
        valid_loader = kettle.validloader
    else:
        train_loader = kettle.trainloader
        valid_loader = kettle.validloader

    for batch, (inputs, labels, ids) in enumerate(tqdm(train_loader)):
        # Prep Mini-Batch
        optimizer.zero_grad()

        # Transfer to GPU
        inputs = inputs.to(**kettle.setup)
        labels = labels.to(dtype=torch.long, device=kettle.setup['device'], non_blocking=NON_BLOCKING)


        # #### Add poison pattern to data #### #
        if poison_delta is not None:
            poison_slices, batch_positions = kettle.lookup_poison_indices(ids)
            if len(batch_positions) > 0:
                inputs[batch_positions] += poison_delta[poison_slices].to(**kettle.setup)


        # Add data augmentation
        inputs = kettle.augment(inputs)


        # Switch into training mode
        list(model.children())[-1].train() if model.frozen else model.train()

        def criterion(outputs, labels):
            loss = loss_fn(outputs, labels)
            predictions = torch.argmax(outputs.data, dim=1)
            correct_preds = (predictions == labels).sum().item()
            return loss, correct_preds

        # Do normal model updates, possibly on modified inputs
        outputs = model(inputs)
        loss, preds = criterion(outputs, labels)
        correct_preds += preds

        total_preds += labels.shape[0]
        differentiable_params = [p for p in model.parameters() if p.requires_grad]

        loss.backward()
        epoch_loss += loss.item()

        optimizer.step()

    scheduler.step()

    if epoch % 10 == 0 or epoch == (10 - 1):
        predictions, valid_loss = run_validation(model, loss_fn, valid_loader,
                                                 kettle.poison_setup['intended_class'],
                                                 kettle.poison_setup['target_class'],
                                                 kettle.setup) # num_workers改为4在这里崩了，说明这里是cpu
        target_acc, target_loss, target_clean_acc, target_clean_loss = check_targets(
            model, loss_fn, kettle.targetset, kettle.poison_setup['intended_class'],
            kettle.poison_setup['target_class'],
            kettle.setup)
    else:
        predictions, valid_loss = None, None
        target_acc, target_loss, target_clean_acc, target_clean_loss = [None] * 4

    current_lr = optimizer.param_groups[0]['lr']
    print_and_save_stats(epoch, stats, current_lr, epoch_loss / (batch + 1), correct_preds / total_preds,
                         predictions, valid_loss,
                         target_acc, target_loss, target_clean_acc, target_clean_loss)


def run_validation(model, criterion, dataloader, intended_class, original_class, setup, dryrun=False):
    """Get accuracy of model relative to dataloader.

    Hint: The validation numbers in "base" and "target" explicitely reference the first label in intended_class and
    the first label in original_class.
    """
    model.eval()
    intended_class = torch.tensor(intended_class).to(device=setup['device'], dtype=torch.long)
    original_class = torch.tensor(original_class).to(device=setup['device'], dtype=torch.long)
    predictions = defaultdict(lambda: dict(correct=0, total=0))

    loss = 0

    with torch.no_grad():
        for i, (inputs, targets, _) in enumerate(dataloader):
            inputs = inputs.to(**setup)
            targets = targets.to(device=setup['device'], dtype=torch.long, non_blocking=NON_BLOCKING)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            loss += criterion(outputs, targets).item()
            predictions['all']['total'] += targets.shape[0]
            predictions['all']['correct'] += (predicted == targets).sum().item()

            predictions['base']['total'] += (targets == intended_class[0]).sum().item()
            predictions['base']['correct'] += (predicted == targets)[targets == intended_class[0]].sum().item()

            predictions['target']['total'] += (targets == original_class).sum().item()
            predictions['target']['correct'] += (predicted == targets)[targets == original_class].sum().item()

            if dryrun:
                break

    for key in predictions.keys():
        if predictions[key]['total'] > 0:
            predictions[key]['avg'] = predictions[key]['correct'] / predictions[key]['total']
        else:
            predictions[key]['avg'] = float('nan')

    loss_avg = loss / (i + 1)
    return predictions, loss_avg

def check_targets(model, criterion, targetset, intended_class, original_class, setup):
    """Get accuracy and loss for all targets on their intended class."""
    model.eval()
    if len(targetset) > 0:

        target_images = torch.stack([data[0] for data in targetset]).to(**setup)
        intended_labels = torch.tensor(intended_class).to(device=setup['device'], dtype=torch.long)
        original_labels = torch.stack([torch.as_tensor(data[1], device=setup['device'], dtype=torch.long) for data in targetset])
        with torch.no_grad():
            outputs = model(target_images)
            predictions = torch.argmax(outputs, dim=1)

            loss_intended = criterion(outputs, intended_labels)
            accuracy_intended = (predictions == intended_labels).sum().float() / predictions.size(0)
            loss_clean = criterion(outputs, original_labels)
            predictions_clean = torch.argmax(outputs, dim=1)
            accuracy_clean = (predictions == original_labels).sum().float() / predictions.size(0)

            # print(f'Raw softmax output is {torch.softmax(outputs, dim=1)}, intended: {intended_class}')

        return accuracy_intended.item(), loss_intended.item(), accuracy_clean.item(), loss_clean.item()
    else:
        return 0, 0, 0, 0


def _split_data(inputs, labels, target_selection='sep-half'):
    """Split data for meta update steps and other defenses."""
    batch_size = inputs.shape[0]
    #  shuffle/sep-half/sep-1/sep-10
    if target_selection == 'shuffle':
        shuffle = torch.randperm(batch_size, device=inputs.device)
        temp_targets = inputs[shuffle].detach().clone()
        temp_true_labels = labels[shuffle].clone()
        temp_fake_label = labels
    elif target_selection == 'michael':  # experimental option
        temp_targets = inputs
        temp_true_labels = labels
        bins = torch.bincount(labels, minlength=10)  # this is only valid for CIFAR-10
        least_appearing_label = bins.argmin()
        temp_fake_label = least_appearing_label.repeat(batch_size)
    elif target_selection == 'sep-half':
        temp_targets, inputs = inputs[:batch_size // 2], inputs[batch_size // 2:]
        temp_true_labels, labels = labels[:batch_size // 2], labels[batch_size // 2:]
        temp_fake_label = labels.mode(keepdim=True)[0].repeat(batch_size // 2)
    elif target_selection == 'sep-1':
        temp_targets, inputs = inputs[0:1], inputs[1:]
        temp_true_labels, labels = labels[0:1], labels[1:]
        temp_fake_label = labels.mode(keepdim=True)[0]
    elif target_selection == 'sep-10':
        temp_targets, inputs = inputs[0:10], inputs[10:]
        temp_true_labels, labels = labels[0:10], labels[10:]
        temp_fake_label = labels.mode(keepdim=True)[0].repeat(10)
    elif 'sep-p' in target_selection:
        p = int(target_selection.split('sep-p')[1])
        p_actual = int(p * batch_size / 128)
        if p_actual > batch_size or p_actual < 1:
            raise ValueError(f'Invalid sep-p option given with p={p}. Should be p in [1, 128], '
                             f'which will be scaled to the current batch size.')
        inputs, temp_targets, = inputs[0:p_actual], inputs[p_actual:]
        labels, temp_true_labels = labels[0:p_actual], labels[p_actual:]
        temp_fake_label = labels.mode(keepdim=True)[0].repeat(batch_size - p_actual)

    else:
        raise ValueError(f'Invalid selection strategy {target_selection}.')
    return temp_targets, inputs, temp_true_labels, labels, temp_fake_label


def get_optimizers(model, opti_name, lr, weight_decay, epochs):
    """Construct optimizer as given in defs."""
    # For transfer learning, we restrict the parameters to be optimized.
    # This filter should only trigger if self.args.scenario == 'transfer'
    optimized_parameters = filter(lambda p: p.requires_grad, model.parameters())

    if opti_name == 'SGD':
        optimizer = torch.optim.SGD(optimized_parameters, lr=lr, momentum=0.9,
                                    weight_decay=weight_decay, nesterov=True)
    elif opti_name == 'SGD-basic':
        optimizer = torch.optim.SGD(optimized_parameters, lr=lr, momentum=0.0,
                                    weight_decay=weight_decay, nesterov=False)
    elif opti_name == 'AdamW':
        optimizer = torch.optim.AdamW(optimized_parameters, lr=lr, weight_decay=weight_decay)
    elif opti_name == 'Adam':
        optimizer = torch.optim.Adam(optimized_parameters, lr=lr, weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=[epochs // 2.667, epochs // 1.6,
                                                                     epochs // 1.142], gamma=0.1)

        # Example: epochs=160 leads to drops at 60, 100, 140.
    return optimizer, scheduler
