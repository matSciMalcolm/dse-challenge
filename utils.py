from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import KFold, GroupKFold
from sklearn.utils import resample
from itertools import accumulate
from uuid import uuid1
from collections import Counter
import pandas
import numpy


class Result():
    def __init__(self,
                 model,
                 round_num,
                 data_size,
                 accuracy,
                 accuracy_std,
                 f1,
                 f1_std,
                 recall,
                 recall_std,
                 precision,
                 precision_std):

        self.model = model
        self.round_num = round_num
        self. data_size = data_size
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
        with open(f'results/{self.model}_{self.uid}_{self.round_num}_report.txt', 'w+') as f:
            f.write(f'\nModel Type: {self.model}')
            f.write(f'\nRound Number: {self.round_num}')
            f.write(f'\nTrain Rows: {self.data_size}')
            f.write(f'\nModel accuracy: {self.accuracy}')
            f.write(f'\nModel accuracy_std: {self.accuracy_std}')
            f.write(f'\nModel f1: {self.f1}')
            f.write(f'\nModel f1_std: {self.f1_std}')
            f.write(f'\nModel recall: {self.recall}')
            f.write(f'\nModel recall_std: {self.recall_std}')
            f.write(f'\nModel precision: {self.precision}')
            f.write(f'\nModel precision_std: {self.precision_std}')


def run_k_folds(model, inputs, outputs, groups, sampling=False, ramp=False, vector=False, splits=5):
    name = type(model).__name__
    results = []
    gkf = GroupKFold(n_splits=10)

    if ramp:
        chunks = [numpy.split(a, splits) for a in (inputs, outputs, groups)]
        batchs = [list(accumulate(chunk, stack)) for chunk in chunks]
        rounds = list(map(list, zip(*batchs)))
    else:
        rounds = [[inputs, outputs, groups]]

    for i, data_round in enumerate(rounds):
        accuracies = []
        f1s = []
        inputs, outputs, groups = data_round
        for train, test in gkf.split(inputs, outputs, groups=groups):
            # Data Preprocessing
            if sampling:
                train = oversample(train, outputs)

            model.fit(inputs[train], outputs[train])
            prediction = model.predict(inputs[test])
            accuracies.append(accuracy_score(outputs[test], prediction))
            f1s.append(precision_recall_fscore_support(
                outputs[test], prediction, average='binary'))

        f1_df = pandas.DataFrame(
            f1s, columns=['precision', 'recall', 'f1', 'support'])
        res = Result(model=name,
                     round_num=i,
                     data_size=len(inputs),
                     accuracy=numpy.mean(accuracies),
                     accuracy_std=numpy.std(accuracies),
                     f1=f1_df['f1'].mean(),
                     f1_std=f1_df['f1'].std(),
                     recall=f1_df['recall'].mean(),
                     recall_std=f1_df['recall'].std(),
                     precision=f1_df['precision'].mean(),
                     precision_std=f1_df['precision'].std())
        res.save()
        results.append(res)
    return results


def oversample(train, outputs):
    majority_class, minority_class = Counter(outputs[train]).most_common()
    minority = train[numpy.where(outputs[train] == minority_class[0])[0]]
    majority = train[numpy.where(outputs[train] == majority_class[0])[0]]
    minority_upsampled = resample(minority,
                                  replace=True,
                                  n_samples=len(majority),
                                  random_state=8)
    return numpy.append(majority, minority_upsampled)


report_column_labels = ['type',
                        'data_size',
                        'accuracy',
                        'accuracy_std',
                        'f1',
                        'f1_std',
                        'recall',
                        'recall_std',
                        'precision',
                        'precision_std']


def stack(a, b):
    return numpy.concatenate((a, b))


def compile_data(results):
    return [[r.model,
             r.data_size,
             r.accuracy,
             r.accuracy_std,
             r.f1,
             r.f1_std,
             r.recall,
             r.recall_std,
             r.precision,
             r.precision_std] for r in results]
