import pandas
import numpy
import typing
from pprint import pprint
from functools import partial
from pymatgen import Composition


class DataManager():
    def __init__(self, load_path, save_path):
        self.load_path = load_path
        self.save_path = save_path
        self.data = None
        self.num_records = None
        self.groups = None
        self.featurized_data = None
        self.outputs = None
        self.sto_dict = {"0.1": 1,
                         "0.2": 1,
                         "0.3": 1,
                         "0.4": 2,
                         "0.5": 1,
                         "0.6": 3,
                         "0.7": 2,
                         "0.8": 4,
                         "0.9": 9}

    def load(self):
        self.data = pandas.read_csv(self.load_path)
        self.num_records = len(self.data.index)
        pprint(f"Loaded {self.num_records} records.")

    def sample_data(self, sample_size: int = 1000):
        if not self.data.empty:
            self.data = self.data.sample(sample_size)
            self.data.reset_index(drop=True, inplace=True)
            self.num_records = len(self.data.index)


    def get_pymatgen_composition(self):
        if not self.data.empty:
            def _get_composition(c):
                """Attempt to parse composition, return None if failed"""
                try:
                    return Composition(c)
                except:
                    return None

            self.data['composition'] = self.data['formula'].apply(
                _get_composition)

    def remove_noble_gasses(self):
        if not self.data.empty:
            def _check_nobility(row):
                comp = row['composition']
                return comp.contains_element_type('noble_gas')

            self.data['noble'] = self.data.apply(_check_nobility, axis=1)
            self.data = self.data[self.data['noble'] == False]
            self.data.reset_index(drop=True, inplace=True)

    def remove_features(self):
        if 'composition' in self.data.columns:
            self.data = self.data[['formula',
                                   'composition', 'group', 'stable']]

    def compute_formula(self):
        self.data['formula'] = self.data['formulaA'] + self.data['formulaB']

    def save_to_csv(self):
        pass

    def to_binary_classes(self):

        def _vec_to_stability(row: pandas.Series, cols: list) -> pandas.Series:
            vec = eval(row['stabilityVec'])
            for element, col in zip(vec, cols):
                row[col] = int(element)
            return row
        
        def _frac_to_sto(frac: float):
            return sto
        
        def _row_to_formula(row: pandas.Series):
            w = float(row['weight_fraction_element_b'])
            a = row['formulaA']
            b = row['formulaB']

            if w == 0.0:
                return a
            elif w == 1.0:
                return b
            else:
                wa = round(1.0-w,1)
                wa = self.sto_dict.get(str(wa), wa)
                w = self.sto_dict.get(str(w), w)

                def _compute_formula(wa):
                    return f"{a}{wa}{b}{w}"
                return _compute_formula(wa)

        cols = ['{}'.format(i/10) for i in range(11)]
        _vtf = partial(_vec_to_stability, cols=cols)

        self.groups = numpy.arange(self.num_records)
        self.data['group'] = self.groups
        original_cols = self.data.columns

        self.data = self.data.apply(_vtf, axis=1)
        self.data = self.data.melt(
            id_vars=original_cols, var_name='weight_fraction_element_b', value_name='stable')
        self.data['formula'] = self.data.apply(_row_to_formula, axis=1)
        self.data = self.data.drop_duplicates('formula')
