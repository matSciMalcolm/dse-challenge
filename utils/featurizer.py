#### Standard Libraries ####
import typing

#### Third-party Libraries ####
import pandas
import numpy
from matminer.featurizers import composition as cf
from matminer.featurizers.conversions import CompositionToOxidComposition
from matminer.featurizers.composition import OxidationStates
from matminer.featurizers.base import MultipleFeaturizer


class Featurizer():
    """[Encapsulates featurization opperations]
    ...

    Attributes
    ----------
    feature_set {[str]} -- [A list of acccepted key-words (standard,
    cmpd_energy, electronegativity, energy_a. energy_b) to set which
    MatMiner compositional features to use] (default: {['standard']})

    mp_api_key {str} -- [Your materials project API key] (default: {None})

    Methods
    -------
    featurize(data) -- [Apply the compositional featurizers to data]
    """

    def __init__(self, feature_set: ['str'] = ['standard'],
                 mp_api_key: str = None):
        """
        Keyword Arguments:
            feature_set {[str]} -- [A list of acccepted key-words (standard,
            cmpd_energy, electronegativity, energy_a. energy_b) to set which
            MatMiner compositional features to use] (default: {['standard']})
            mp_api_key {str} -- [Your materials project API key] (default: {None})
        """

        self.feature_set = feature_set
        self.mp_api_key = mp_api_key
        self.featurizers = {'standard': [cf.Stoichiometry(),
                                         cf.ElementProperty.from_preset(
                                             "magpie"),
                                         cf.ValenceOrbital(props=['avg']),
                                         cf.IonProperty(fast=True)],
                            'cmpd_energy': [cf.CohesiveEnergy(mapi_key=self.mp_api_key)],
                            'electronegativity': [cf.OxidationStates(),
                                                  cf.ElectronegativityDiff()],
                            'energy_a': [],
                            'energy_b': []}

    def featurize(self, data: pandas.DataFrame) -> numpy.array:
        """[Apply the compositional featurizers to data]

        Arguments:
            data {pandas.DataFrame} -- [A dataframe with pymatgen Composition
            objects under the column 'composition']

        Returns:
            numpy.array -- [Featurized data]
        """
        impute = False
        if 'electronegtivity' in self.feature_set:
            data = CompositionToOxidComposition(return_original_on_error=True,
                                                overwrite_data=True).\
                featurize_dataframe(data,
                                    'composition',
                                    ignore_errors=True)

        if any(c in self.feature_set for c in ('cmpd_energy', 'energy_a',
                                               'energy_b', 'electronegativity')):
            impute = True

        features = []
        for feature in self.feature_set:
            features.extend(self.featurizers.get(feature))

        f = MultipleFeaturizer(features)
        X = numpy.array(f.featurize_many(
            data['composition'], ignore_errors=True))

        if 'energy_a' in self.feature_set:
            f = MultipleFeaturizer(
                [cf.CohesiveEnergy(mapi_key=self.mp_api_key)])
            A = numpy.array(f.featurize_many(
                data['formulaA'], ignore_errors=True))
            X = numpy.concatenate((X, A), axis=1)

        if 'energy_b' in self.feature_set:
            f = MultipleFeaturizer(
                [cf.CohesiveEnergy(mapi_key=self.mp_api_key)])
            B = numpy.array(f.featurize_many(
                data['formulaB'], ignore_errors=True))
            X = numpy.concatenate((X, B), axis=1)

        if impute:
            return numpy.nan_to_num(X)
        else:
            return X
