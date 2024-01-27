# fes_tools
## A simple Python library for analyzing and extracting information from free energy surfaces from molecular simulation.

### Required Packages:
- NumPy
- Pandas
- SciPy
- Matplotlib
- ImageMagick
- jupyter (just for the notebook example)

A YML file is provided:
conda env create --file fes_tools.yml -n "fes_tools"

All can be installed via conda from scratch
conda install pandas matplotlib imagemagick jupyter scipy

### Installation:
From the directory with setup.py: pip install -e ./ --user

### Usage

import fes_tools into your python scripts and use the class based structure to 
analyze your simualtions.

fes_tools can extract the ∆G and of a reaction and the transition state energy
of currently 1 and 2 dimension free energy surfaces, as well as plot them.

This can be done via the fes class for single surfaces or many sequential surfaces.

An example Jupyter Notebook is provided.

