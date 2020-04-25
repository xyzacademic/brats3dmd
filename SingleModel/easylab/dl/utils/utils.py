from easylab.dl.metrics import get_dict



evaluation_method = get_dict()


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name):
        self.name = name
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


def evaluation(predict, label, method):
    if method not in evaluation_method:
        raise KeyError("{} not found in default evaluation method!".format(method))
    return evaluation_method[method](predict, label)



