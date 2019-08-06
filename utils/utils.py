#### Standard Libraries ####
import typing
from itertools import accumulate
from uuid import uuid1
from collections import Counter

#### Third-party Libraries ####
import pandas
import numpy
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import KFold, GroupKFold
from sklearn.utils import resample

# Useful labels for building reports
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


class Result():
    """[A class for storing the performance on a k-fold model evaluation.]
    ...

    Attributes
    ----------
    model {str} -- [The name of the type of model evaluated]
    round_num {int} -- [The round of evaluation during a data ramp]
    data_size {int} -- [The amount of data used to train for the given round]
    accuracy {float} -- [Classification accuracy]
    accuracy_std {float} -- [Standard deviation in classification accuracy]
    f1 {float} -- [F1 score]
    f1_std {float} -- [Standard deviation in the F1 score]
    recall {float} -- [The recall score]
    recall_std {float} -- [Standard deviation in the recall score]
    precision {float} -- [The precision score]
    precision_std {float} -- [Standard deviation in the precision score]
    uid {str} -- [Unique ID for the evaluation]

    Methods
    -------
    save(self) -- [Save the results of a k-folds evaluation to a file]
    """

    def __init__(self,
                 model: str,
                 round_num: int,
                 data_size: int,
                 accuracy: float,
                 accuracy_std: float,
                 f1: float,
                 f1_std: float,
                 recall: float,
                 recall_std: float,
                 precision: float,
                 precision_std: float):
        """
        Arguments:
            model {str} -- [The name of the type of model evaluated]
            round_num {int} -- [The round of evaluation during a data ramp]
            data_size {int} -- [The amount of data used to train for the given
            round]
            accuracy {float} -- [Classification accuracy]
            accuracy_std {float} -- [Standard deviation in classification
            accuracy]
            f1 {float} -- [F1 score]
            f1_std {float} -- [Standard deviation in the F1 score]
            recall {float} -- [The recall score]
            recall_std {float} -- [Standard deviation in the recall score]
            precision {float} -- [The precision score]
            precision_std {float} -- [Standard deviation in the precision score]
        """

        self.model = model
        self.round_num = round_num
        self.data_size = data_size
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
        """[Save the results of a k-folds evaluation to a file]
        """
        with open(f'results/{self.model}_{self.uid}_{self.round_num}_report.txt',
                  'w+') as f:
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


def run_k_folds(model, inputs: numpy.array, outputs: numpy.array,
                groups: pandas.Series, sampling: bool = False, ramp: bool = False,
                splits: int = 5) -> Result:
    """[Runs group 10-fold cross validation to determine the accuracy, precision,
    and recall of a classification model while accomodating super sampling and
    data ramping]

    Arguments:
        model -- [A machine learning model similiar to scikit-learn]
        inputs {numpy.array} -- [The array of input values]
        outputs {numpy.array} -- [The array of output labels]
        groups {pandas.Series} -- [The groups corresponding to the inputs]

    Keyword Arguments:
        sampling {bool} -- [Should th eminority output
        class be super sampled] (default: {False})
        ramp {bool} -- [Should the data be added in small chunks ] (default: {False})
        splits {int} -- [The number of evenly sized chunks to divde the training
        data into when running a data ramp] (default: {5})

    Returns:
        Result -- [A Result object containing the scores and metadata for the 
        cross validation run]
    """

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


def oversample(train: numpy.array, outputs: numpy.array) -> numpy.array:
    """[Deterines the minority class and then draws samples until the class
    representation is balanced]

    Arguments:
        train {numpy.array} -- [Indicies of the input for the training fold]
        outputs {numpy.array} -- [The output data]

    Returns:
        numpy.array -- [Supersampled indicies of the input data corresponding
        to the training fold]
    """
    majority_class, minority_class = Counter(outputs[train]).most_common()
    minority = train[numpy.where(outputs[train] == minority_class[0])[0]]
    majority = train[numpy.where(outputs[train] == majority_class[0])[0]]
    minority_upsampled = resample(minority,
                                  replace=True,
                                  n_samples=len(majority),
                                  random_state=8)
    return numpy.append(majority, minority_upsampled)


def stack(a: numpy.array, b: numpy.array) -> numpy.array:
    """[Anonymous function for stacking data]

    Arguments:
        a {numpy.array} -- [Previous array]
        b {numpy.array} -- [Array to stack]

    Returns:
        numpy.array -- [Stacked array]
    """
    return numpy.concatenate((a, b))


def compile_data(results: List) -> List:
    """[Converts attributes of a list of Result objects into lists]

    Arguments:
        results {Result} -- [description]

    Returns:
        List -- [List]
    """

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
