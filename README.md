# Material Predictions
This repository contains Code for my submission to the Citrine Informatice data solutions engineer challenge. The goal of this challenge is to predict the stability vector of a system of two elements. In order to achieve this I have transformed the training data into a binary classification scheme to better transfer the complexity of the output to the input. 

## How to use this repository
1. Install requirments with `pip install -r requirements.txt`, for best results start with a new anaconda python 3 environement
2. Install the package itself by running `pip install .`
3. Store your [Materials Project](https://materialsproject.org/) API key as the first line in a file named `mp_api_key.txt` under the configuration directory
4. Store training data as `training_data.csv` in the `data` directory

## Where to start
The repo is structured as follows:
* `data` - Training and test data.
* `results` - Results of group cross validation studies.
* `configuration` - Configuration files like your Materials Project API key.
* `notebooks` - The core demonstration and code to use this repo.
* `utils` - Useful classes and tools for machine learning and visualization.
* `models` - Saved machine learning models.

To start, navigate to the `notebooks` directory. Here you will find a progression of Jupyter notebooks that explore how we chose the correct model for stability. `0_interactive_predict` gives the user access to an interactive periodic table where they can choose two elements and get a predicted stability vector in real time. `1_evaluate_models` is a powerful tool for running group k-folds cross validation on multiple models in parallel an was used to test different modeling approached. `2_compare_results` describes the experiments we ran with notebook 1 and showcases the results of different approaches. Finally `3_predict_stability_vector` is used to label the test data set with predicted stability vectors. The results of which can be found in `data` as `test_csv_labeled.csv`. If your interested in learning more about my approach, check out the slide deck.

![alt text](https://github.com/matSciMalcolm/dse-challenge/blob/master/data/example.png "Interactive periodic table")

