import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="dse-challenge",
    version="0.0.1",
    author="Malcolm Davidson",
    author_email="mdavidson@citrine.io",
    description="Notebooks to predict the stability of binary chemical compounds.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/matSciMalcolm/dse-challenge",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",


        "Operating System :: OS Independent",],
    install_requires=[
        "scikit-learn>=0.20.3",
        "numpy>=1.16.2",
        "pandas>=0.24.2",
        "matminer>=0.5.8",
        "pymatgen>=2019.6.20",
        "plotly>=3.10.0",
        "mendeleev>=0.4.5",
        "matplotlib>=3.0.3"])