from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import KFold, GroupKFold
from sklearn.utils import resample
from uuid import uuid1
import pandas
import numpy


class Result():
    def __init__(self,
                 model,
                 accuracy,
                 accuracy_std,
                 f1,
                 f1_std,
                 recall,
                 recall_std,
                 precision,
                 precision_std):

        self.model = model
        self.accuracy = accuracy
        self.accuracy_std = accuracy_std
        self.f1 = f1
        self.f1_std = f1_std
        self.recall = recall
        self.recall_std = recall_std
        self.precision = precision
        self.precision_std = precision_std
        self.uid = str(uuid1())

    def save(self):
        with open(f'results/{self.model}_{self.uid}_report.txt', 'w+') as f:
            f.write(f'\nModel Type: {self.model}')
            f.write(f'\nModel accuracy: {self.accuracy}')
            f.write(f'\nModel accuracy_std: {self.accuracy_std}')
            f.write(f'\nModel f1: {self.f1}')
            f.write(f'\nModel f1_std: {self.f1_std}')
            f.write(f'\nModel recall: {self.recall}')
            f.write(f'\nModel recall_std: {self.recall_std}')
            f.write(f'\nModel precision: {self.precision}')
            f.write(f'\nModel precision_std: {self.precision_std}')


def run_k_folds(model, inputs, outputs, groups, sampling = False, vector = False):
    name = type(model).__name__
    accuracies = []
    f1s = []
    cnt = 1

    gkf = GroupKFold(n_splits=10)
    for train, test in gkf.split(inputs, outputs, groups=groups):
        # Data Preprocessing
        if sampling:
            train = oversample(train, outputs)
            
        model.fit(inputs[train], outputs[train])
        prediction = model.predict(inputs[test])
        accuracies.append(accuracy_score(outputs[test], prediction))
        f1s.append(precision_recall_fscore_support(
            outputs[test], prediction, average='binary'))
        cnt += 1

    f1_df = pandas.DataFrame(
        f1s, columns=['precision', 'recall', 'f1', 'support'])
    res = Result(model=name,
                 accuracy=numpy.mean(accuracies),
                 accuracy_std=numpy.std(accuracies),
                 f1=f1_df['f1'].mean(),
                 f1_std=f1_df['f1'].std(),
                 recall=f1_df['recall'].mean(),
                 recall_std=f1_df['recall'].std(),
                 precision=f1_df['precision'].mean(),
                 precision_std=f1_df['precision'].std())
    res.save()
    return res


def oversample(train, outputs):
    unstable = numpy.where(outputs[train] == 0, )[0]
    stable = numpy.where(outputs[train] == 1)[0]
    stable_upsampled = resample(stable,
                              replace=True,
                              n_samples=len(unstable),
                              random_state=8)                          
    return numpy.append(unstable, stable_upsampled)


report_column_labels = ['type',
        'accuracy',
        'accuracy_std',
        'f1',
        'f1_std',
        'recall',
        'recall_std',
        'precision',
        'precision_std']


def compile_data(results):
    return [[r.model,
             r.accuracy,
             r.accuracy_std,
             r.f1,
             r.f1_std,
             r.recall,
             r.recall_std,
             r.precision,
             r.precision_std] for r in results]