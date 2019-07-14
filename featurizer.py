import pandas
import numpy
from typing import List
from matminer.featurizers import composition as cf
from matminer.featurizers.conversions import CompositionToOxidComposition
from matminer.featurizers.composition import OxidationStates
from matminer.featurizers.base import MultipleFeaturizer


class Featurizer():
    def __init__(self, feature_set: ['str'] = ['standard'], mp_api_key: str = None):
        self.feature_set = feature_set
        self.mp_api_key = mp_api_key

        self.featurizers = {'standard': [cf.Stoichiometry(),
                                         cf.ElementProperty.from_preset(
                                             "magpie"),
                                         cf.ValenceOrbital(props=['avg']),
                                         cf.IonProperty(fast=True)],
                            'energy': [cf.CohesiveEnergy(mapi_key=self.mp_api_key)],
                            'electronegativity': [cf.OxidationStates(),
                                                  cf.ElectronegativityDiff()]}

    def featurize(self, data: pandas.DataFrame) -> pandas.DataFrame:
        impute = False
        if 'electronegtivity' in self.feature_set:
            data = CompositionToOxidComposition(return_original_on_error=True,
                                                overwrite_data=True).featurize_dataframe(data,
                                                                                         'composition',
                                                                                         ignore_errors=True)
        if any(c in self.feature_set for c in ('energy', 'electronegativity')):
            impute = True

        features = []
        for feature in self.feature_set:
            features.extend(self.featurizers.get(feature))

        f = MultipleFeaturizer(features)
        X = numpy.array(f.featurize_many(data['composition'], ignore_errors=True))

        if impute:
            return numpy.nan_to_num(X)
        else:
            return X
