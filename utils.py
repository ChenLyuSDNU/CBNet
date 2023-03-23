import shutil
import torch
import sys
import os
import json
import numpy as np
from config import config
from torch import nn
import torch.nn.functional as F


def save_checkpoint(state, is_best, fold):
    filename = config.weights + config.model_name + os.sep + str(fold) + os.sep + "_checkpoint.pth.tar"
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename,
                        config.best_models + config.model_name + os.sep + str(fold) + os.sep + 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""

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


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 3 epochs"""
    lr = config.lr * (0.1 ** (epoch // 3))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def schedule(current_epoch, current_lrs, **logs):
    lrs = [1e-3, 1e-4, 0.5e-4, 1e-5, 0.5e-5]
    epochs = [0, 1, 6, 8, 12]
    for lr, epoch in zip(lrs, epochs):
        if current_epoch >= epoch:
            current_lrs[5] = lr
            if current_epoch >= 2:
                current_lrs[4] = lr * 1
                current_lrs[3] = lr * 1
                current_lrs[2] = lr * 1
                current_lrs[1] = lr * 1
                current_lrs[0] = lr * 0.1
    return current_lrs


def eval(output, target):
    correct = 0
    total = 0
    classnum = 3
    target_num = torch.zeros((1, classnum))
    predict_num = torch.zeros((1, classnum))
    acc_num = torch.zeros((1, classnum))
    _, predicted = torch.max(output.data, 1)
    total += target.size(0)
    correct += predicted.eq(target.data).cpu().sum()
    pre_mask = torch.zeros(output.size()).scatter_(1, predicted.cpu().view(-1, 1), 1.)
    predict_num += pre_mask.sum(0)
    tar_mask = torch.zeros(output.size()).scatter_(1, target.data.cpu().view(-1, 1), 1.)
    target_num += tar_mask.sum(0)
    acc_mask = pre_mask * tar_mask
    acc_num += acc_mask.sum(0)
    recall = acc_num / target_num
    precision = acc_num / predict_num
    F1 = 2 * recall * precision / (recall + precision)
    accuracy = acc_num.sum(1) / target_num.sum(1)
    # 精度调整
    recall = (recall.numpy()[0] * 100).round(3)
    precision = (precision.numpy()[0] * 100).round(3)
    F1 = (F1.numpy()[0] * 100).round(3)
    accuracy = (accuracy.numpy()[0] * 100).round(3)
    # 打印格式方便复制
    print('recall', " ".join('%s' % id for id in recall))
    print('precision', " ".join('%s' % id for id in precision))
    print('F1', " ".join('%s' % id for id in F1))
    print('accuracy', accuracy)
    return recall, precision, F1


import time


# 计算模型推理时间函数
def measure_inference_speed(model, data, max_iter=200, log_interval=50):
    model.eval()

    # the first several iterations may be very slow so skip them
    num_warmup = 5
    pure_inf_time = 0
    fps = 0

    # benchmark with 2000 image and take the average
    for i in range(max_iter):

        torch.cuda.synchronize()
        start_time = time.perf_counter()

        with torch.no_grad():
            model(*data)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time

        if i >= num_warmup:
            pure_inf_time += elapsed
            if (i + 1) % log_interval == 0:
                fps = (i + 1 - num_warmup) / pure_inf_time
                print(
                    f'Done image [{i + 1:<3}/ {max_iter}], '
                    f'fps: {fps:.1f} img / s, '
                    f'times per image: {1000 / fps:.1f} ms / img',
                    flush=True)

        if (i + 1) == max_iter:
            fps = (i + 1 - num_warmup) / pure_inf_time
            print(
                f'Overall fps: {fps:.1f} img / s, '
                f'times per image: {1000 / fps:.1f} ms / img',
                flush=True)
            break
    return fps


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        classnum = 3
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout  # stdout
        self.file = None

    def open(self, file, mode=None):
        if mode is None: mode = 'w'
        self.file = open(file, mode)

    def write(self, message, is_terminal=1, is_file=1):
        if '\r' in message: is_file = 0

        if is_terminal == 1:
            self.terminal.write(message)
            self.terminal.flush()
            # time.sleep(1)

        if is_file == 1:
            self.file.write(message)
            self.file.flush()

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


def get_learning_rate(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]

    # assert(len(lr)==1) #we support only one param_group
    lr = lr[0]

    return lr


def time_to_str(t, mode='min'):
    if mode == 'min':
        t = int(t) / 60
        hr = t // 60
        min = t % 60
        return '%2d hr %02d min' % (hr, min)

    elif mode == 'sec':
        t = int(t)
        min = t // 60
        sec = t % 60
        return '%2d min %02d sec' % (min, sec)


    else:
        raise NotImplementedError


class FocalLoss(nn.Module):

    def __init__(self, focusing_param=2, balance_param=0.25):
        super(FocalLoss, self).__init__()

        self.focusing_param = focusing_param
        self.balance_param = balance_param

    def forward(self, output, target):
        cross_entropy = F.cross_entropy(output, target)
        cross_entropy_log = torch.log(cross_entropy)
        logpt = - F.cross_entropy(output, target)
        pt = torch.exp(logpt)

        focal_loss = -((1 - pt) ** self.focusing_param) * logpt

        balanced_focal_loss = self.balance_param * focal_loss

        return balanced_focal_loss


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)
