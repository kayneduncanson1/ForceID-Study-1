import time
import os
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch
from Losses import get_dist_and_masks
from datetime import datetime


class AverageMeter(object):

    def __init__(self):

        self.reset()

    def reset(self):

        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):

        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=20, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):

        if self.best_loss == None:
            self.best_loss = val_loss

        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss

            # reset counter if validation loss improves
            self.counter = 0

        elif self.best_loss - val_loss < self.min_delta:

            self.counter += 1
            # print(f"INFO: Early stopping counter {self.counter} of {self.patience}")

            if self.counter >= self.patience:

                # print('INFO: Early stopping')
                self.early_stop = True


"""The only difference between each train_val and each test func is the input args that are passed into the call to
criterion."""


# The base class for the main experiments:
def train_val(mod, data_loader_tr, data_loader_va, labels_va, criterion, sample_count_min, opt, epochs):

    early_stopping = EarlyStopping()
    since_2 = time.time()
    tr_acc_hist = []
    tr_loss_hist = []
    va_acc_hist = []
    va_loss_hist = []
    best_acc_tr = 0
    best_acc_va = 0

    for epoch in range(1, epochs + 1):

        for phase in ['train', 'val']:

            if phase == 'train':

                losses = AverageMeter()
                accs = AverageMeter()
                mod.cuda().train()

                for batch_idx, (inputs, labs) in enumerate(data_loader_tr):

                    inputs, labs = inputs.cuda(), labs.cuda()
                    opt.zero_grad()

                    with torch.set_grad_enabled(True):

                        embs = mod(inputs)

                        dist, mask_pos, mask_neg = get_dist_and_masks(embs, labs)

                        # Note that min_sample_count doesn't actually get used in the training phase but is a required
                        # argument of criterion:
                        loss, acc = criterion(dist, mask_pos, mask_neg, phase, sample_count_min)

                        embs = None
                        dist = None
                        mask_pos = None
                        mask_neg = None

                        # (Loss is None in the case that there are no valid triplets in the mini-batch):
                        if loss is not None:

                            loss.backward()
                            opt.step()
                            losses.update(loss.detach().cpu().numpy().item())
                            accs.update(acc)

                tr_loss_hist.append(losses.avg)
                tr_acc_hist.append(accs.avg)

                if accs.avg > best_acc_tr:

                    best_acc_tr = accs.avg

            else:

                mod.eval()

                embs = None

                for batch_idx, inputs in enumerate(data_loader_va):

                    inputs = inputs.cuda()
                    opt.zero_grad()

                    with torch.set_grad_enabled(False):

                        outputs = mod(inputs)

                        embs = torch.cat((embs, outputs), dim=0) if embs is not None else outputs

                # Remove data from outputs variable to save memory while calculating loss and acc:
                outputs = None

                # labels_va is not loaded using the data loader above but included as parameter. Convert to cuda here:
                dist, mask_pos, mask_neg = get_dist_and_masks(embs, labels_va.cuda())

                loss, acc = criterion(dist, mask_pos, mask_neg, phase, sample_count_min)

                dist = None
                mask_pos = None
                mask_neg = None

                va_loss_hist.append(loss.detach().cpu().numpy().item())
                va_acc_hist.append(acc)

                if acc > best_acc_va:

                    best_acc_va = acc
                    checkpoint = {'model_state_dict': mod.state_dict(), 'optimizer_state_dict': opt.state_dict()}

        early_stopping(loss)

        if early_stopping.early_stop:

            break

    time_elapsed = time.time() - since_2

    return time_elapsed, tr_acc_hist, tr_loss_hist, va_acc_hist, va_loss_hist, checkpoint['model_state_dict'],\
        checkpoint['optimizer_state_dict'], embs


# The class used for reducing validation and test set ap and an distance matrices to contain a pre-defined no. of
# samples per ID (n_samples_per_ID) to compare with each anchor. This class was used to validate the performance
# evaluation setting (see Supp Material - Section I):
def train_val_supp_pt1(mod, data_loader_tr, data_loader_va, labels_va, criterion, sample_count_min, n_samples_per_ID,
                       opt, epochs):

    early_stopping = EarlyStopping()
    since_2 = time.time()
    tr_acc_hist = []
    tr_loss_hist = []
    va_acc_hist = []
    va_loss_hist = []
    best_acc_tr = 0
    best_acc_va = 0

    for epoch in range(1, epochs + 1):

        for phase in ['train', 'val']:

            if phase == 'train':

                losses = AverageMeter()
                accs = AverageMeter()
                mod.cuda().train()

                for batch_idx, (inputs, labs) in enumerate(data_loader_tr):

                    inputs, labs = inputs.cuda(), labs.cuda()
                    opt.zero_grad()

                    with torch.set_grad_enabled(True):

                        embs = mod(inputs)

                        dist, mask_pos, mask_neg = get_dist_and_masks(embs, labs)

                        # Note that min_sample_count and n_samples_per_ID don't actually get used in the training phase
                        # but are required arguments of criterion:
                        loss, acc = criterion(dist, mask_pos, mask_neg, phase, sample_count_min, n_samples_per_ID)

                        embs = None
                        dist = None
                        mask_pos = None
                        mask_neg = None

                        # (Loss is None in the case that there are no valid triplets in the mini-batch):
                        if loss is not None:
                            loss.backward()
                            opt.step()
                            losses.update(loss.detach().cpu().numpy().item())
                            accs.update(acc)

                tr_loss_hist.append(losses.avg)
                tr_acc_hist.append(accs.avg)

                if accs.avg > best_acc_tr:
                    best_acc_tr = accs.avg

            else:

                mod.eval()

                embs = None

                for batch_idx, inputs in enumerate(data_loader_va):
                    inputs = inputs.cuda()
                    opt.zero_grad()

                    with torch.set_grad_enabled(False):
                        outputs = mod(inputs)

                        embs = torch.cat((embs, outputs), dim=0) if embs is not None else outputs

                # Remove data from outputs variable to save memory while calculating loss and acc:
                outputs = None

                # labels_va is not loaded using the data loader above but included as parameter. Convert to cuda here:
                dist, mask_pos, mask_neg = get_dist_and_masks(embs, labels_va.cuda())

                loss, acc = criterion(dist, mask_pos, mask_neg, phase, sample_count_min, n_samples_per_ID)

                dist = None
                mask_pos = None
                mask_neg = None

                va_loss_hist.append(loss.detach().cpu().numpy().item())
                va_acc_hist.append(acc)

                if acc > best_acc_va:
                    best_acc_va = acc
                    checkpoint = {'model_state_dict': mod.state_dict(), 'optimizer_state_dict': opt.state_dict()}

        early_stopping(loss)

        if early_stopping.early_stop:
            break

    time_elapsed = time.time() - since_2

    return time_elapsed, tr_acc_hist, tr_loss_hist, va_acc_hist, va_loss_hist, checkpoint['model_state_dict'], \
           checkpoint['optimizer_state_dict'], embs


# The class used when not reducing the no. of samples per ID to compare with each anchor in validation and test
# sets. This is the 'All' condition in Supp Material - Section I. Note the indices_s1 and indices_s2
# input arguments:"""
def train_val_supp_pt2(mod, data_loader_tr, data_loader_va, labels_va, criterion, indices_s1, indices_s2, opt, epochs):

    early_stopping = EarlyStopping()
    since_2 = time.time()
    tr_acc_hist = []
    tr_loss_hist = []
    va_acc_hist = []
    va_loss_hist = []
    best_acc_tr = 0
    best_acc_va = 0

    for epoch in range(1, epochs + 1):

        for phase in ['train', 'val']:

            if phase == 'train':

                losses = AverageMeter()
                accs = AverageMeter()
                mod.cuda().train()

                for batch_idx, (inputs, labs) in enumerate(data_loader_tr):

                    inputs, labs = inputs.cuda(), labs.cuda()
                    opt.zero_grad()

                    with torch.set_grad_enabled(True):

                        embs = mod(inputs)

                        dist, mask_pos, mask_neg = get_dist_and_masks(embs, labs)
                        # Note that indices_s1 and indices_s2 don't actually get used in the
                        # training phase but are required arguments of criterion:
                        loss, acc = criterion(dist, mask_pos, mask_neg, phase, indices_s1, indices_s2)

                        embs = None
                        dist = None
                        mask_pos = None
                        mask_neg = None

                        if loss is not None:

                            loss.backward()
                            opt.step()
                            losses.update(loss.detach().cpu().numpy().item())
                            accs.update(acc)

                tr_loss_hist.append(losses.avg)
                tr_acc_hist.append(accs.avg)

                if accs.avg > best_acc_tr:

                    best_acc_tr = accs.avg

            else:

                mod.eval()

                embs = None

                for batch_idx, inputs in enumerate(data_loader_va):

                    inputs = inputs.cuda()
                    opt.zero_grad()

                    with torch.set_grad_enabled(False):

                        outputs = mod(inputs)
                        embs = torch.cat((embs, outputs), dim=0) if embs is not None else outputs

                # Remove data from outputs to save memory while calculating loss and acc:
                outputs = None

                # labels_va is not loaded using the data loader above but included as parameter. Convert to cuda here:
                dist, mask_pos, mask_neg = get_dist_and_masks(embs, labels_va.cuda())

                loss, acc = criterion(dist, mask_pos, mask_neg, phase, indices_s1, indices_s2)

                dist = None
                mask_pos = None
                mask_neg = None

                va_loss_hist.append(loss.detach().cpu().numpy().item())
                va_acc_hist.append(acc)

                if acc > best_acc_va:

                    best_acc_va = acc
                    checkpoint = {'model_state_dict': mod.state_dict(), 'optimizer_state_dict': opt.state_dict()}

        early_stopping(loss)

        if early_stopping.early_stop:

            break

    time_elapsed = time.time() - since_2

    return time_elapsed, tr_acc_hist, tr_loss_hist, va_acc_hist, va_loss_hist, checkpoint['model_state_dict'],\
        checkpoint['optimizer_state_dict'], embs


# The base class for the main experiments:
def test(mod, opt, data_loader_te, labels_te, criterion, sample_count_min):

    mod.eval()

    embs = None

    for batch_idx, inputs in enumerate(data_loader_te):

        inputs = inputs.cuda()
        opt.zero_grad()

        with torch.set_grad_enabled(False):

            outputs = mod(inputs)
            embs = torch.cat((embs, outputs), dim=0) if embs is not None else outputs

    # Remove data from outputs to save memory while calculating loss and acc:
    outputs = None

    # labels_te is not loaded using the data loader above but included as parameter. Convert to cuda here:
    dist, mask_pos, mask_neg = get_dist_and_masks(embs, labels_te.cuda())

    # Use phase='val' so that the test acc is calculated in the same way as the val acc:
    loss, acc = criterion(dist, mask_pos, mask_neg, 'val', sample_count_min)

    dist = None
    mask_pos = None
    mask_neg = None

    return loss.cpu().numpy().item(), acc, embs


# The class used for reducing validation and test set ap and an distance matrices to contain a pre-defined no. of
# samples per ID (n_samples_per_ID) to compare with each anchor. This class was used to validate the performance
# evaluation setting (see Supp Material - Section I):
def test_supp_pt1(mod, opt, data_loader_te, labels_te, criterion, sample_count_min, n_samples_per_ID):

    mod.eval()

    embs = None

    for batch_idx, inputs in enumerate(data_loader_te):

        inputs = inputs.cuda()
        opt.zero_grad()

        with torch.set_grad_enabled(False):

            outputs = mod(inputs)
            embs = torch.cat((embs, outputs), dim=0) if embs is not None else outputs

    # Remove data from outputs to save memory while calculating loss and acc:
    outputs = None

    # labels_te is not loaded using the data loader above but included as parameter. Convert to cuda here:
    dist, mask_pos, mask_neg = get_dist_and_masks(embs, labels_te.cuda())

    # Use phase='val' so that the test acc is calculated in the same way as the val acc:
    loss, acc = criterion(dist, mask_pos, mask_neg, 'val', sample_count_min, n_samples_per_ID)

    dist = None
    mask_pos = None
    mask_neg = None

    return loss.cpu().numpy().item(), acc, embs


# The class used when not reducing the no. of samples per ID to compare with each anchor in validation and test
# sets. This is the 'All' condition in Supp Material - Section I. Note the indices_s1 and indices_2
# input arguments:"""
def test_supp_pt2(mod, opt, data_loader_te, labels_te, criterion, indices_s1, indices_s2):

    mod.eval()

    embs = None

    for batch_idx, inputs in enumerate(data_loader_te):

        inputs = inputs.cuda()
        opt.zero_grad()

        with torch.set_grad_enabled(False):

            outputs = mod(inputs)
            embs = torch.cat((embs, outputs), dim=0) if embs is not None else outputs

    # Remove data from outputs to save memory while calculating loss and acc:
    outputs = None

    # labels_va is not loaded using the data loader above but included as parameter. Convert to cuda here:
    dist, mask_pos, mask_neg = get_dist_and_masks(embs, labels_te.cuda())

    # Use phase='val' so that the test acc is calculated in the same way as the val acc:
    loss, acc = criterion(dist, mask_pos, mask_neg, 'val', indices_s1, indices_s2)

    dist = None
    mask_pos = None
    mask_neg = None

    return loss.cpu().numpy().item(), acc, embs

