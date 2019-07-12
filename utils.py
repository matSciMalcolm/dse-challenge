from pymatgen import Composition
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import KFold
from uuid import uuid1
import pandas
import numpy


def get_compostion(c):
    """Attempt to parse composition, return None if failed"""

    try:
        return Composition(c)
    except:
        return None


def check_nobility(row):
    comp = row['composition']
    return comp.contains_element_type('noble_gas')


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


def run_k_folds(model, inputs, outputs):
    name = type(model).__name__
    accuracies = []
    f1s = []
    cnt = 1

    for train, test in KFold(n_splits=10, shuffle=True, random_state=8).split(inputs):
        # Data Preprocessing
        model.fit(inputs[train], outputs[train])
        prediction = model.predict(inputs[test])
        accuracies.append(accuracy_score(outputs[test], prediction))
        f1s.append(precision_recall_fscore_support(outputs[test], prediction, average='binary'))
        cnt += 1

    f1_df = pandas.DataFrame(f1s, columns=['precision', 'recall', 'f1', 'support'])
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
