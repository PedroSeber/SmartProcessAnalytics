# Smart Process Analytics

Smart Process Analytics (SPA) is a Python software for predictive modeling. Since 2022, this version has been updated by Pedro Seber after being forked from [the original version](https://github.com/vickysun5/SmartProcessAnalytics). This fork is different enough from the original version that it should be considered its own thing.

To run SPA on your computer, simply download the source code of the [most recent release](https://github.com/PedroSeber/SmartProcessAnalytics/releases). Unzip that folder somewhere convenient, open a terminal in that folder, (optionally) create a new conda environment or activate your conda environment of choice, and run `pip install -e .` (note the dot after -e). SPA should then be usable after `import SPA` and calling the `SPA.main_SPA()` function. If you are having issues installing ace-cream, comment its line out in the [setup.py](setup.py) file and try again. Most of SPA will work without ace-cream.

[SPA.py](Code-SPA/SPA.py) comes with default hyperparameters for its models, but all hyperparameters are customizable by the user. To learn how to do so, please read its documentation. You may also check the [Examples](Examples) folder and the [README](Examples/README.md) within.

The major files in SPA are:<br>
1. [SPA.py](Code-SPA/SPA.py): the main file and what should be called by the user. Calls the files below depending on what inputs have been passed by the user or the properties of the data.<br>
2. [cv\_final.py](Code-SPA/cv_final.py): performs cross-validation (or IC calculations) to automatically determine the best hyperparameters. Also trains the final model after validation.<br>
3. [regression\_models.py](Code-SPA/regression_models.py): called multiple times by [cv\_final.py](Code-SPA/cv_final.py); runs a model once based on one combination of hyperparameters.<br>
4. [dataset\_property\_new.py](Code-SPA/dataset_property_new.py): functions for data interrogation to determine whether the data exhibit nonlinearity, multicollinearity, and/or dynamics. Mostly ignored if the user selects a model architecture(s) manually.

A typical run of [SPA.py](Code-SPA/SPA.py) automatically calls [cv\_final.py](Code-SPA/cv_final.py) once to determine the best hyperparameters and return the best model. For each hyperparameter, [cv\_final.py](Code-SPA/cv_final.py) automatically calls [regression\_models.py](Code-SPA/regression_models.py) once per hyperparameter combination for validation. If the user has not supplied a model type (or types), [SPA.py](Code-SPA/SPA.py) also calls [dataset\_property\_new.py](Code-SPA/dataset_property_new.py) to determine the most adequate model(s) for the data.

The final result is stored in the `selected_model` and `fitting_result` variables returned by [SPA.py](Code-SPA/SPA.py). It is also saved as .json and .p files.

## Citing
This version of SPA has been the subject of multiple publications. If you have used SPA, please cite the following works (depending on what was used). Bibtex-formatted citations are available in [citation.bib](citation.bib).

| Publication name | Please cite if... |
| :----------: | :----: |
| [LCEN: A Novel Feature Selection Algorithm for Nonlinear, Interpretable Machine Learning Models](https://arxiv.org/abs/2402.17120) | You used LCEN for any task.
| [Improving N-Glycosylation and Biopharmaceutical Production Predictions Using AutoML-Built Residual Hybrid Models](https://doi.org/10.1101/2024.08.27.609988) | You used MLPs or RNNs generated by SPA, even if not for a residual hybrid model.

Please contact Pedro Seber (pseber[at]mit{dot}edu) and/or Richard Braatz (braatz[at]mit{dot}edu) for any inquiries.
