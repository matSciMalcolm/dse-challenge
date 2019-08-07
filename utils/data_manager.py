#### Standard Libraries ####
import typing
from typing import List
from pprint import pprint
from functools import partial

#### Third-party Libraries ####
import pandas
import numpy
from pymatgen import Composition


class DataManager():
    """[A class to help track and manipulate chemical data to be used for
    machine learning]
    ...

    Attributes
    ----------
    load_path {str} -- [The path to the training data]
    save_path {str} -- [The path to save data]
    data {pandas.DataFrame} -- [Unfeaturized data]
    original_data {pandas.DataFrame} -- [Data as loaded from file]
    num_records {int} -- [The number of data poinds]
    groups {numpy.array} -- [The group id's for chemical systems.
    Used during cross validation]
    featurized_data {numpy.array}} -- [Stores an array of featurized data when
    used with a featurizer]
    outputs {pandas.Series} -- [The output labels for data]
    labeled_data {pandas.DataFrame} -- [Test data labeled with stability vectors]

    Methods
    -------
    load() -- [Read in data from a csv]
    sample_data(sample_size=1000) -- [Convert data to a random sample of the 
    total data]
    loads(elements: list) -- [Read in data from a list of elements]
    get_pymatgen_composition() -- [Add a 'composition' column to data 
    consisting of pymatgen composition objects]
    remove_noble_gasses() -- [Remove any noble gasses from composition]
    remove_features(data) -- [Remove any columns beyond ['formula', 'formulaA',
    'formulaB', 'composition', 'group', 'stable'] from data]
    compute_formula() -- [Add a 'formula' column to data by concatenating
    formulaA and formulaB]
    to_binary_classes() -- [Convert training data to a binary classification
    representation in place of a stability vector]
    convert_inputs() -- [Convert test data to systems of integer chemical
    formulas corresponding to elements of the stability vector]
    validate_data() -- [Check if the input columns Formula A and B have valid
    elements]
    """

    def __init__(self, load_path: str=None, save_path: str=None):
        """
        Arguments:
            load_path {str} -- [The path to the training data]
            save_path {str} -- [The path to save data]
        """
        self.load_path = load_path
        self.save_path = save_path
        self.data = None
        self.original_data = None
        self.num_records = None
        self.groups = None
        self.featurized_data = None
        self.outputs = None
        self.labeled_data = None
        self.sto_dict = {"0.1": 1,
                         "0.2": 1,
                         "0.3": 1,
                         "0.4": 2,
                         "0.5": 1,
                         "0.6": 3,
                         "0.7": 2,
                         "0.8": 4,
                         "0.9": 9}
        self.formula_index = ["{a}",
                              "{a}9{b}1",
                              "{a}4{b}1",
                              "{a}2{b}1",
                              "{a}3{b}2",
                              "{a}1{b}1",
                              "{a}2{b}3",
                              "{a}1{b}2",
                              "{a}1{b}4",
                              "{a}1{b}9",
                              "{a}",
                              "system",
                              "formulaA",
                              "formulaB"]

    def load(self):
        """[Read in data from a csv]
        """

        self.data = pandas.read_csv(self.load_path)
        self.original_data = self.data
        self.num_records = len(self.data.index)
        pprint(f"Loaded {self.num_records} records.")

    def loads(self, elements: List):
        """[Read in data from a tuple]
        """
        
        self.data = pandas.DataFrame(elements, columns=['formulaA', 'formulaB'])
        self.original_data = self.data
        self.num_records = len(self.data.index)

    def sample_data(self, sample_size: int = 1000):
        """[Convert data to a random sample of the total data]

        Keyword Arguments:
            sample_size {int} -- [The number of records to 
            randomly chose] (default: {1000})
        """

        if not self.data.empty:
            self.data = self.data.sample(sample_size)
            self.data.reset_index(drop=True, inplace=True)
            self.num_records = len(self.data.index)

    def get_pymatgen_composition(self):
        """[summAdd a 'composition' column to data consisting of pymatgen
        composition objectsary]
        """

        if not self.data.empty:
            def _get_composition(c):
                """Attempt to parse composition, return None if failed"""
                try:
                    return Composition(c)
                except:
                    return None

            self.data['composition'] = self.data['formula'].apply(
                _get_composition)
            self.data['formulaA'] = self.data['formulaA'].apply(
                _get_composition)
            self.data['formulaB'] = self.data['formulaB'].apply(
                _get_composition)

    def remove_noble_gasses(self):
        """[Remove any noble gasses from composition]
        """

        if not self.data.empty:
            def _check_nobility(row):
                comp = row['composition']
                return comp.contains_element_type('noble_gas')

            self.data['noble'] = self.data.apply(_check_nobility, axis=1)
            self.data = self.data[self.data['noble'] == False]
            self.data.reset_index(drop=True, inplace=True)

    def remove_features(self):
        """[Remove any columns beyond ['formula', 'formulaA', 'formulaB',
        'composition', 'group', 'stable'] from data]
        """

        if 'composition' in self.data.columns:
            self.data = self.data[['formula',
                                   'formulaA',
                                   'formulaB',
                                   'composition',
                                   'group',
                                   'stable']]

    def compute_formula(self):
        """[Add a 'formula' column to data by concatenating formulaA and
        formulaB]
        """

        self.data['formula'] = self.data['formulaA'] + self.data['formulaB']

    def to_binary_classes(self):
        """[Convert training data to a binary classification representation in
        place of a stability vector]
        """

        def _vec_to_stability(row: pandas.Series, cols: list) -> pandas.Series:
            vec = eval(row['stabilityVec'])
            for element, col in zip(vec, cols):
                row[col] = int(element)
            return row

        def _row_to_formula(row: pandas.Series):
            w = float(row['weight_fraction_element_b'])
            a = row['formulaA']
            b = row['formulaB']

            if w == 0.0:
                return a
            elif w == 1.0:
                return b
            else:
                wa = round(1.0-w, 1)
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
            id_vars=original_cols, var_name='weight_fraction_element_b',
            value_name='stable')
        self.data['formula'] = self.data.apply(_row_to_formula, axis=1)
        self.data = self.data.drop_duplicates('formula')

    def convert_inputs(self):
        """[Convert test data to systems of integer chemical formulas
        corresponding to elements of the stability vector]
        """

        def _convert(row):
            a = row['formulaA']
            b = row['formulaB']
            system = a+b
            _formula_templates = [f"{a}",
                                  f"{a}9{b}1",
                                  f"{a}4{b}1",
                                  f"{a}2{b}1",
                                  f"{a}3{b}2",
                                  f"{a}1{b}1",
                                  f"{a}2{b}3",
                                  f"{a}1{b}2",
                                  f"{a}1{b}4",
                                  f"{a}1{b}9",
                                  f"{b}",
                                  system,
                                  f"{a}",
                                  f"{b}", ]

            formulas = pandas.Series(
                _formula_templates, index=self.formula_index)
            return(formulas)

        self.data = self.data.apply(_convert,
                                    axis=1,
                                    result_type='expand').\
            melt(id_vars=['system', 'formulaA', 'formulaB'],
                 value_name="formula").drop(['variable'], axis=1)

    def validate_data(self):
        """[Check if the input columns Formula A and B have valid elements]
        """
        # Validate data - move to data manager
        test_col_1 = self.data.iloc[:, 0].apply(lambda x: Composition(x).valid)
        test_col_2 = self.data.iloc[:, 1].apply(lambda x: Composition(x).valid)
        if not all(test_col_1) or not all(test_col_2):
            pprint("Invalid element in data")
        else:
            pprint("All input elements are valid")

    def binary_to_vec(self):
        """[Converts predicted binary labels into a stability vector]
        """

        def _binary_to_vec(df: pandas.DataFrame):
            return df.tolist()

        self.labeled_data = self.original_data
        self.labeled_data['stabilityVec'] = self.data.groupby('system')\
            .agg(_binary_to_vec)['stable'].values
