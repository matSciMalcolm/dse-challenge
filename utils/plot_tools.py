#### Standard Libraries ####
import os
from collections import Counter

# Third-party Libraries
import plotly
import pickle
import numpy
import pandas
import plotly.plotly as py
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from mendeleev import get_table, element
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

#### Local Libraries ####
#from data_manager import DataManager
from utils.featurizer import Featurizer
from utils.data_manager import DataManager


def plot_results(df: pandas.DataFrame, title: str) -> plt.Axes:
    """[summary]

    Arguments:
        df {pandas.DataFrame} -- [description]
        title {str} -- [description]

    Returns:
        plt.Axes -- [description]
    """
    err = df[['accuracy_std', 'f1_std']]
    err.columns = ['accuracy', 'f1']
    ax = df[['accuracy', 'f1']].plot.barh(
        y=['accuracy', 'f1'], xerr=err, figsize=(8, 4))
    ax.set_yticklabels(df['type'])
    ax.set_xlim((0, 1))
    ax.grid(False)
    ax.legend(loc=(1.04, 0.85))
    ax.set_title(str(title))
    return ax


def plot_feature_results(df: pandas.DataFrame, mp_ax: plt.Axes,
                         title: str) -> plt.Axes:
    """[summary]

    Arguments:
        df {pandas.DataFrame} -- [description]
        mp_ax {plt.Axes} -- [description]
        title {str} -- [description]

    Returns:
        plt.Axes -- [description]
    """

    # Sort the data for easier and consistent comparrison
    df = df.sort_values('order')

    err = df[['accuracy_std', 'f1_std']]
    err.columns = ['accuracy', 'f1']
    ax = df[['accuracy', 'f1']].plot.barh(
        y=['accuracy', 'f1'], xerr=err, ax=mp_ax)
    ax.set_yticklabels(df['feature_set'])
    ax.set_xlim((0, 1))
    ax.set_title(str(title))
    ax.legend().set_visible(False)
    ax.grid(False)

    return ax


def plot_ramp_results(df: pandas.DataFrame, mp_ax: plt.Axes,
                      title: str) -> plt.Axes:
    """

    Arguments:
        df {pandas.DataFrame} -- [description]
        mp_ax {plt.Axes} -- [description]
        title {str} -- [description]

    Returns:
        plt.Axes -- [description]
    """

    # Sort the data for easier and consistent comparrison
    df = df.sort_values('data_size')

    err = df[['accuracy_std', 'f1_std']]
    err.columns = ['accuracy', 'f1']
    ax = df[['accuracy', 'f1']].plot.barh(
        y=['accuracy', 'f1'], xerr=err, ax=mp_ax)
    ax.set_yticklabels(df['data_size'])
    ax.set_xlim((0, 1))
    ax.set_title(str(title))
    ax.legend().set_visible(False)
    ax.grid(False)

    return ax


def interactive_pt(mp_api_key_path):
    feature_set = ['standard', 'cmpd_energy']
    color_dict = {'s': 'red', 'p': 'blue', 'd': 'green', 'f': 'yellow'}
    load_model_path = os.path.join('..', 'models', 'rfc.sav')

    init_notebook_mode(connected=True)

    # Initialize
    with open(mp_api_key_path, 'r') as f:
        mp_api_key = f.readline().rstrip()
    dm = DataManager()
    fz = Featurizer(feature_set=feature_set, mp_api_key=mp_api_key)
    with open(load_model_path, "rb") as f:
        model = pickle.load(f)

    # Set up periodic table data
    pt = get_table('elements')
    pt_grid = pt[['name', 'symbol', 'group_id',
                  'period', 'block']].astype(int, errors='ignore')
    pt_grid['color'] = pt_grid['block'].replace(color_dict)
    # add plot info to la/ac
    mask_1 = (pt_grid.block == 'f') & (
        pt_grid.index >= 58) & (pt_grid.index <= 71)
    mask_2 = (pt_grid.block == 'f') & (
        pt_grid.index >= 90) & (pt_grid.index <= 103)
    pt_grid.loc[mask_1, 'group_id'] = numpy.arange(3, 16)
    pt_grid.loc[mask_2, 'group_id'] = numpy.arange(3, 16)
    pt_grid.loc[mask_1, 'period'] = 9
    pt_grid.loc[mask_2, 'period'] = 10

    data = [go.Scatter(x=pt_grid['group_id'],
                       y=pt_grid['period'],
                       mode='markers+text',
                       text=pt_grid['symbol'],
                       hoverinfo="text",
                       showlegend=False,
                       hovertext=pt_grid['name'],
                       textfont=dict(
        color='black'),
        marker=dict(
        size=20,
        color=pt_grid['color'],
        opacity=0.5,
        line=dict(
            width=1,
            color='grey'),
        symbol=1)),
        go.Scatter(x=[7],
                   y=[2],
                   showlegend=False,
                   mode="lines+text",
                   text="",
                   hovertext=False,
                   textposition="middle center")]

    layout = go.Layout(
        yaxis=dict(
            autorange='reversed',
            visible=False),
        xaxis=dict(
            visible=False))

    f = go.FigureWidget(data=data, layout=layout)

    scatter = f.data[0]
    label = f.data[1]
    scatter.marker.color = pt_grid['color']
    scatter.marker.size = [20] * 118
    f.layout.hovermode = 'closest'

    # create our callback function
    def update_point(trace, points, selector):
        c = list(scatter.marker.color)
        s = list(scatter.marker.size)

        '''
        count = Counter(c)
        if count['#bae2be'] >= 2:
            scatter.marker.color = pt_grid['color']
            scatter.marker.size = [20] * 118
            label.text = ""

        else:
            for i in points.point_inds:
                c[i] = '#bae2be'
                s[i] = 25
                scatter.marker.color = c
                scatter.marker.size = s
        '''
        # Record clicks
        for i in points.point_inds:
                c[i] = '#bae2be'
                s[i] = 25
                scatter.marker.color = c
                scatter.marker.size = s

        count = Counter(c)
        if count['#bae2be'] == 2:
            elements = [i for i, e in enumerate(c, 1) if e == '#bae2be']
            A = element(elements[0]).symbol
            B = element(elements[1]).symbol
            dm.loads([[A, B]])

            # Featurize
            dm.convert_inputs()
            dm.get_pymatgen_composition()
            dm.featurized_data = fz.featurize(dm.data)

            # Predict
            dm.data['stable'] = model.predict(dm.featurized_data)
            dm.binary_to_vec()
            vec = dm.labeled_data['stabilityVec'].tolist()[0]

            # Label
            label.text = f"{A}{B}   {vec}"
            scatter.marker.color = pt_grid['color']
            scatter.marker.size = [20] * 118
            elements = []

    scatter.on_click(update_point)
    return f

# f
