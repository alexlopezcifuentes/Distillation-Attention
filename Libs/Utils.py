import numpy as np
import torch
import shutil
import os
import yaml

"""
Using a DCT-driven Loss in Attention-based Knowledge-Distillation for Scene Recognition

Utils.py
Bunch of utils to compute things.

Fully developed by Anonymous Code Author.
"""

class AverageMeter(object):
    """
    Class to store instant values, accumulated and average of measures
    """
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.sum2 = 0
        self.count = 0
        self.std = 0
        self.list_val = []

    def update(self, val, n=1):
        self.val = val
        self.list_val.extend([val] * n)
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

        # Compute STD
        a = [np.power(x - self.avg, 2) for x in self.list_val]
        self.std = np.sqrt(np.mean(a))


class EpochMeter(object):
    """
    Class to store epoch lists with values. Similar to Average Meter but for epochs. It enables to use it in Precision mode or Loss Mode.
    """
    def __init__(self, mode=''):
        self.mode = mode
        self.classification = []
        if mode.lower() == 'loss':
            self.total = []
            self.total_up = []
            self.total_down = []
            self.distill = []

    def update(self, values):
        self.classification.append(values['classification'].avg)

        if self.mode.lower() == 'loss':
            self.total.append(values['total'].avg)
            self.total_up.append(values['total'].avg + values['total'].std)
            self.total_down.append(values['total'].avg - values['total'].std)
            self.distill.append(values['distillation'].avg)


def accuracy(output, target, topk=(1,)):
    """
    Computes the top-k accuracy between output and target.
    :param output: output vector from the network
    :param target: ground-truth
    :param topk: Top-k results desired, i.e. top1, top5, top10
    :return: vector with accuracy values
    """
    maxk = max(topk)
    batch_size = target.size(0)

    # ------------------------ #
    #        PREDICTION        #
    # ------------------------ #

    _, pred = output.topk(maxk, 1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    # ------------------- #
    #      ACCURACIES     #
    # ------------------- #

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res


def classAccuracy(CM):
    """
    Compute class accuracy based on a given Confusion Matrix
    :param CM: Confusion Matrix
    :return: Class accuracy
    """
    class_acc = list()
    n_classes = CM.shape[0]

    for i in range(n_classes):
        num_GT = np.sum(CM[i, :])
        num_correct_pred = CM[i, i]
        class_acc.append(num_correct_pred / (num_GT + 1e-09))

    return class_acc


def save_checkpoint(state, is_best, path, filename='checkpoint.pth.tar'):
    """
    Funtion to save a model as a checkpoint
    :param state: Dictonary with all the information to save
    :param is_best: Boolean variable that states if the model is the best acheived or not
    :param path: Path to save the models
    :param filename: Filename to save the file.
    :return: None
    """
    torch.save(state, os.path.join(path, 'Models', filename + '_latest.pth.tar'))
    if is_best:
        print('Best model updated.')
        shutil.copyfile(os.path.join(path, 'Models', filename + '_latest.pth.tar'),
                        os.path.join(path, 'Models', filename + '_best.pth.tar'))

        # Summary Stats
        sumary_dict_file = {'TRAIN LOSS': str(round(state['best_loss_train'], 2)),
                            'VALIDATION LOSS': str(round(state['best_loss_val'], 2)),
                            'TRAIN ACCURACY': str(round(state['best_prec_train'], 2)),
                            'VALIDATION ACCURACY': str(round(state['best_prec_val'], 2)),
                            'EPOCH': state['epoch'],
                            'EPOCH TIME': str(round(state['time_per_epoch'], 2)) + ' Minutes',
                            'COMMENTS': state['CONFIG']['MODEL']['COMMENTS'],
                            'MODEL PARAMETERS': str(state['model_parameters']) + ' Millions',
                            'ARCHITECTURE': state['CONFIG']['MODEL']['ARCH'],
                            'DATASET': state['CONFIG']['DATASET']['NAME']}

        # Save new summary stats for the new best model
        with open(os.path.join(path, state['CONFIG']['MODEL']['ARCH'] + ' ' + state['CONFIG']['DATASET']['NAME']
                                     + ' Summary Report.yaml'), 'w') as file:
            yaml.safe_dump(sumary_dict_file, file)

