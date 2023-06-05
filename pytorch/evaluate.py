from sklearn import metrics
from sklearn.metrics import accuracy_score
from pytorch.pytorch_utils import forward
import numpy as np

class Evaluator(object):
    def __init__(self, model):
        """Evaluator.

        Args:
          model: object
        """
        self.model = model
        
    def evaluate(self, data_loader):
        """Forward evaluation data and calculate statistics.

        Args:
          data_loader: object

        Returns:
          statistics: dict, 
              {'average_precision': (classes_num,), 'auc': (classes_num,)}
        """

        # Forward
        output_dict = forward(
            model=self.model, 
            generator=data_loader, 
            return_target=True)

        clipwise_output = output_dict['clipwise_output']    # (audios_num, classes_num)
        target = output_dict['target']    # (audios_num, classes_num)

        average_precision = metrics.average_precision_score(
            target, clipwise_output, average='macro')

        #auc = metrics.roc_auc_score(target, clipwise_output, average=None)

        target_acc = np.argmax(target, axis=1)
        clipwise_output_acc = np.argmax(clipwise_output, axis=1)
        #pred = clipwise_output_acc.max(1, keepdim=True)[1]
        #acc = accuracy_score(target_acc, pred)
        acc =accuracy_score(target_acc, clipwise_output_acc)

        each_acc = []
        for i in range(0, 10):
            a = np.argwhere(target_acc == i)     #6 is a class label
            x = target_acc[a]
            y = clipwise_output_acc[a]
            q = y[:, 0]
            w = x[:, 0]
            e = sum(q == w)
            f = e/len(target_acc)
            each_acc.append(f)

        #statistics = {'average_precision': average_precision, 'auc': auc, 'acc': acc}
        statistics = {'average_precision': average_precision, 'acc': acc, 'each_acc': each_acc, 'target_acc':target}

        return statistics
        #computer the classification of each class

