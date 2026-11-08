import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, StratifiedKFold, RepeatedKFold, ShuffleSplit, TimeSeriesSplit, GroupKFold, GroupShuffleSplit, train_test_split
from sklearn.utils._testing import ignore_warnings
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import confusion_matrix
from scipy.special import softmax
import regression_models as rm
from itertools import product
from joblib import Parallel, delayed
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from itertools import product
from collections import OrderedDict
from time import localtime, time
from os.path import exists as os_exists

def CVpartition(X, y, Type = 'Re_KFold', K = 5, Nr = 10, random_state = 0, group = None):
    """
    Partitions data for cross-validation and bootstrapping.
    Returns a generator with the split data.

    Parameters
    ----------
    model_name : str
        Which model to use
    X, y : Numpy array with shape N x m, N x 1
        Training data predictors and response.
    Type : str, optional, default = 'Re_KFold'
        Which cross validation method to use.
    K : int, optional, default = 5
        Number of folds used in cross validation.
    Nr : int, optional, default = 10
        Number of CV repetitions used when cv_type in {'MC', 'Re_KFold', 'GroupShuffleSplit'}.
    random_state : int, optional, default = 0
        Seed used for the random number generator.
    group : list, optional, default = None
        Group indices for grouped CV methods.
    """
    Type = Type.casefold() # To avoid issues with uppercase/lowercase
    if Type == 'mc':
        CV = ShuffleSplit(n_splits = Nr, test_size = 1/K, random_state = random_state)
        for train_index, val_index in CV.split(X, y):
            yield (X[train_index], y[train_index], X[val_index], y[val_index])
    elif Type == 'single':
        X, X_test, y, y_test = train_test_split(X, y, test_size = 1/K, random_state = random_state)
        yield (X, y, X_test, y_test)
    elif Type == 'kfold':
        CV = KFold(n_splits = int(K), shuffle = True, random_state = random_state)
        for train_index, val_index in CV.split(X, y):
            yield (X[train_index], y[train_index], X[val_index], y[val_index])
    elif Type == 'stratifiedkfold':
        CV = StratifiedKFold(n_splits = int(K), shuffle = True, random_state = random_state)
        for train_index, val_index in CV.split(X, y):
            yield (X[train_index], y[train_index], X[val_index], y[val_index])
    elif Type == 're_kfold':
        CV = RepeatedKFold(n_splits = int(K), n_repeats = Nr, random_state = random_state)
        for train_index, val_index in CV.split(X, y):
            yield (X[train_index], y[train_index], X[val_index], y[val_index])
    elif Type == 'timeseries':
        TS = TimeSeriesSplit(n_splits = int(K))
        for train_index, val_index in TS.split(X):
            yield (X[train_index], y[train_index], X[val_index], y[val_index])
    elif Type == 'single_group':
        label = np.unique(group)
        num = int(len(label)/K)
        final_list = np.squeeze(group == label[0])
        for i in range(1, num):
            final_list = final_list | np.squeeze(group == label[i])
        yield(X[~final_list], y[~final_list], X[final_list], y[final_list])
    elif Type == 'group':
        label = np.unique(group)
        for i in range(len(label)):
            yield(X[np.squeeze(group != label[i])], y[np.squeeze(group != label[i])], X[np.squeeze(group == label[i])], y[np.squeeze(group == label[i])])
    elif Type == 'group_no_extrapolation':
        label = np.unique(group)
        for i in range(len(label)):
            if min(label) < label[i] and label[i] < max(label):
                yield(X[np.squeeze(group != label[i])], y[np.squeeze(group != label[i])], X[np.squeeze(group == label[i])], y[np.squeeze(group == label[i])])
    elif Type == 'groupkfold':
        gkf = GroupKFold(n_splits = int(K))
        for train_index, val_index in gkf.split(X, y, groups = group):
            yield (X[train_index], y[train_index], X[val_index], y[val_index])
    elif Type == 'groupshufflesplit':
        gss = GroupShuffleSplit(n_splits = int(Nr), test_size = 1 / K, random_state = random_state)
        for train_index, val_index in gss.split(X, y, groups = group):
            yield (X[train_index], y[train_index], X[val_index], y[val_index])
    elif Type == 'no_cv':
        yield (X, y, X, y)
    elif Type == 'single_ordered':
        num = X.shape[0]
        yield (X[:num-round(X.shape[0]*1/K):], y[:num-round(X.shape[0]*1/K):], X[num-round(X.shape[0]*1/K):], y[num-round(X.shape[0]*1/K):])
    else:
        raise ValueError(f'{Type} is not a valid CV type.')

def CV_mse(model_name, X, y, X_test, y_test, X_unscaled = None, y_unscaled = None, cv_type = 'Re_KFold', K_fold = 5, Nr = 10, eps = 1e-4, group = None, **kwargs):
    """
    Determines the best hyperparameters using MSE based on information criteria.
    Also returns MSE and yhat data for the chosen model.

    Parameters
    ----------
    model_name : str
        Which model to use
    X, y : Numpy array with shape N x m, N x 1
        Training data predictors and response.
    X_test, y_test : Numpy array with shape N_test x m, N_test x 1
        Testing data predictors and response.
    X_unscaled, y_unscaled : Numpy array with shape N x m, N x 1
        The unscaled values of X and y, used during CV to avoid validation set leakage
        Used when model_name != 'LCEN', as it already uses unscaled inputs
    cv_type : str, optional, default = None
        Which cross validation method to use.
    K_fold : int, optional, default = 5
        Number of folds used in cross validation.
    Nr : int, optional, default = 10
        Number of CV repetitions used when cv_type in {'MC', 'Re_KFold', 'GroupShuffleSplit'}.
    eps : float, optional, default = 1e-4
        Tolerance. TODO: expand on this.
    **kwargs : dict, optional
        Non-default hyperparameters for model fitting.
    """
    # Setting up some general kwargs; these should exist in kwargs unless the user called this function manually, which is not recommended
    if 'robust_priority' not in kwargs:
        kwargs['robust_priority'] = False
    if 'l1_ratio' not in kwargs or kwargs['l1_ratio'] is None:
        kwargs['l1_ratio'] = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99]
    if 'alpha' not in kwargs: # Unusual scenario, since SPA passes a default kwargs['alpha'] == 20
        kwargs['alpha'] = np.concatenate(([0], np.logspace(-4.3, 0, 20)))
    elif isinstance(kwargs['alpha'], int): # User passed an integer instead of a list of values
        kwargs['alpha'] = np.concatenate( ([0], np.logspace(-4.3, 0, kwargs['alpha'])) )
    if 'classification' not in kwargs:
        kwargs['classification'] = False
    elif kwargs['classification']: # If doing classification, y has class labels, and thus should not be scaled
        kwargs['scale_y'] = False
    if 'class_weight' not in kwargs or kwargs['class_weight'] is None:
        kwargs['class_weight'] = np.ones(len( set(y.squeeze()) ))
    if 'scale_X' not in kwargs:
        kwargs['scale_X'] = True
    if 'scale_y' not in kwargs and not kwargs['classification']:
        kwargs['scale_y'] = True
    if 'verbosity_level' not in kwargs:
        kwargs['verbosity_level'] = 2
    if 'model_random_state' not in kwargs:
        kwargs['model_random_state'] = 0
    if 'cv_random_state' not in kwargs:
        kwargs['cv_random_state'] = 0

    if model_name == 'EN':
        hyperparam_prod = list(product(kwargs['l1_ratio'], kwargs['alpha']))
        MSE_result = np.empty((len(kwargs['alpha']) * len(kwargs['l1_ratio']), K_fold*Nr)) * np.nan
        Var = np.empty((len(kwargs['alpha']) * len(kwargs['l1_ratio']), K_fold*Nr)) * np.nan

        with Parallel(n_jobs = -1) as PAR:
            for counter, (X_train, y_train, X_val, y_val) in enumerate(CVpartition(X_unscaled, y_unscaled, cv_type, K_fold, Nr, kwargs['cv_random_state'], group)):
                # Rescaling here to avoid validation dataset leakage
                if kwargs['scale_X'] and len(X.shape) == 2 and X.shape[1] > 0: # StandardScaler doesn't work with 3D arrays or with arrays that are 2D but empty in one dimension
                    scaler_x = StandardScaler(with_mean=True, with_std=True)
                    scaler_x.fit(X_train)
                    X_train_scale = scaler_x.transform(X_train)
                    X_val_scale = scaler_x.transform(X_val)
                else:
                    X_train_scale = X_train
                    X_val_scale = X_val
                if kwargs['scale_y'] and not kwargs['classification']:
                    scaler_y = StandardScaler(with_mean=True, with_std=True)
                    scaler_y.fit(y_train)
                    y_train_scale = scaler_y.transform(y_train)
                    y_val_scale = scaler_y.transform(y_val)
                else:
                    y_train_scale = y_train
                    y_val_scale = y_val
                temp = PAR(delayed(rm.EN_fitting)(X_train_scale, y_train_scale, X_val_scale, y_val_scale, this_prod[1], this_prod[0], random_state = kwargs['model_random_state'],
                                        classification = kwargs['classification']) for this_prod in hyperparam_prod)
                if not kwargs['classification']:
                    _, temp_var_count, _, MSE_result[:, counter], _, _ = zip(*temp)
                else:
                    _, temp_var_count, _, loss_dicts, _, _ = zip(*temp)
                    MSE_result[:, counter] = [elem['MCC'] for elem in loss_dicts]
                Var[:, counter] = [np.sum(elem != 0) for elem in temp_var_count]

        MSE_mean = np.nanmean(MSE_result, axis = 1)
        if not kwargs['classification']:
            ind = np.nanargmin(MSE_mean) # Minimize the MSE
        else:
            ind = np.nanargmax(MSE_mean) # Maximize the MCC
        if kwargs['robust_priority']:
            MSE_std = np.nanstd(MSE_result, axis = 1)
            MSE_min = MSE_mean[ind]
            Var_num = np.nansum(Var, axis = 1) # Here, Var is n_hyperparams x n_folds, and Var_num is n_hyperparams
            if not kwargs['classification']:
                MSE_bar = MSE_min + MSE_std[ind]
                ind = np.nonzero( Var_num == np.nanmin(Var_num[MSE_mean < MSE_bar]) )[0][0] # Hyperparams with the lowest number of variables but still within one stdev of the best MSE
            else:
                MSE_bar = MSE_min - MSE_std[ind] # Despite the name, MSE_min is more like MCC_max in this context
                ind = np.nonzero( Var_num == np.nanmin(Var_num[MSE_mean > MSE_bar]) )[0][0] # Hyperparams with the lowest number of variables but still within one stdev of the best MCC

        # Hyperparameter setup
        l1_ratio, alpha = hyperparam_prod[ind]
        hyperparams = {'alpha': alpha, 'l1_ratio': l1_ratio}
        # Fit the final model
        EN_model, EN_params, loss_train, loss_test, yhat_train, yhat_test = rm.EN_fitting(X, y, X_test, y_test, alpha, l1_ratio, random_state = kwargs['model_random_state'], classification = kwargs['classification'])
        return(hyperparams, EN_model, EN_params, loss_train, loss_test, yhat_train, yhat_test, MSE_mean[ind])

    if model_name in {'SCAD', 'MCP'}:
        if model_name == 'SCAD':
            hyperparam_prod = kwargs['alpha'] # alpha here is lambda in the original SCAD paper
            hyperparam_keys = ['alpha'] # For the hyperparams dict below
            MSE_result = np.empty((len(kwargs['alpha']), K_fold*Nr)) * np.nan
            Var = np.empty((len(kwargs['alpha']), K_fold*Nr)) * np.nan
        elif model_name == 'MCP':
            if 'MCP_gamma' not in kwargs:
                kwargs['MCP_gamma'] = [1, 1.5, 2, 2.5, 3, 3.5, 4]
            hyperparam_prod = list(product(kwargs['alpha'], kwargs['MCP_gamma']))
            hyperparam_keys = ['alpha', 'gamma'] # For the hyperparams dict below
            MSE_result = np.empty((len(kwargs['alpha']) * len(kwargs['MCP_gamma']), K_fold*Nr)) * np.nan
            Var = np.empty((len(kwargs['alpha']) * len(kwargs['MCP_gamma']), K_fold*Nr)) * np.nan

        with Parallel(n_jobs = -1) as PAR:
            for counter, (X_train, y_train, X_val, y_val) in enumerate(CVpartition(X_unscaled, y_unscaled, cv_type, K_fold, Nr, kwargs['cv_random_state'], group)):
                # Rescaling here to avoid validation dataset leakage
                if kwargs['scale_X'] and len(X.shape) == 2 and X.shape[1] > 0: # StandardScaler doesn't work with 3D arrays or with arrays that are 2D but empty in one dimension
                    scaler_x = StandardScaler(with_mean=True, with_std=True)
                    scaler_x.fit(X_train)
                    X_train_scale = scaler_x.transform(X_train)
                    X_val_scale = scaler_x.transform(X_val)
                else:
                    X_train_scale = X_train
                    X_val_scale = X_val
                if kwargs['scale_y'] and not kwargs['classification']:
                    scaler_y = StandardScaler(with_mean=True, with_std=True)
                    scaler_y.fit(y_train)
                    y_train_scale = scaler_y.transform(y_train)
                    y_val_scale = scaler_y.transform(y_val)
                else:
                    y_train_scale = y_train
                    y_val_scale = y_val
                temp = PAR(delayed(rm.GenLinEstimator_fitting)(X_train_scale, y_train_scale, X_val_scale, y_val_scale, model_name, this_prod, random_state = kwargs['model_random_state'],
                                        classification = kwargs['classification']) for this_prod in hyperparam_prod)
                if not kwargs['classification']:
                    _, temp_var_count, _, MSE_result[:, counter], _, _ = zip(*temp)
                else:
                    _, temp_var_count, _, loss_dicts, _, _ = zip(*temp)
                    MSE_result[:, counter] = [elem['MCC'] for elem in loss_dicts]
                Var[:, counter] = [np.sum(elem != 0) for elem in temp_var_count]

        MSE_mean = np.nanmean(MSE_result, axis = 1)
        if not kwargs['classification']:
            ind = np.nanargmin(MSE_mean) # Minimize the MSE
        else:
            ind = np.nanargmax(MSE_mean) # Maximize the MCC
        if kwargs['robust_priority']:
            MSE_std = np.nanstd(MSE_result, axis = 1)
            MSE_min = MSE_mean[ind]
            Var_num = np.nansum(Var, axis = 1) # Here, Var is n_hyperparams x n_folds, and Var_num is n_hyperparams
            if not kwargs['classification']:
                MSE_bar = MSE_min + MSE_std[ind]
                ind = np.nonzero( Var_num == np.nanmin(Var_num[MSE_mean < MSE_bar]) )[0][0] # Hyperparams with the lowest number of variables but still within one stdev of the best MSE
            else:
                MSE_bar = MSE_min - MSE_std[ind] # Despite the name, MSE_min is more like MCC_max in this context
                ind = np.nonzero( Var_num == np.nanmin(Var_num[MSE_mean > MSE_bar]) )[0][0] # Hyperparams with the lowest number of variables but still within one stdev of the best MCC

        # Hyperparameter setup
        if not '__iter__' in dir(hyperparam_prod[ind]): # hyperparam_prod[ind] is not an interable, likely because it has only one element. We need to turn it into a list
            best_hyper = [hyperparam_prod[ind]]
        else:
            best_hyper = hyperparam_prod[ind]
        hyperparams = {key: value for key, value in zip( hyperparam_keys, best_hyper )}
        # Fit the final model
        GenLin_model, GenLin_params, loss_train, loss_test, yhat_train, yhat_test = rm.GenLinEstimator_fitting(X, y, X_test, y_test, model_name, hyperparam_prod[ind], random_state = kwargs['model_random_state'], classification = kwargs['classification'])
        return(hyperparams, GenLin_model, GenLin_params, loss_train, loss_test, yhat_train, yhat_test, MSE_mean[ind])


    elif model_name == 'SPLS' or model_name == 'PLS':
        if not(cv_type.startswith('Group')) and ('SPLS_K' not in kwargs or kwargs['SPLS_K'] is None): # For non-grouped CV types
            kwargs['SPLS_K'] = np.linspace( 1, min(X.shape[1], int((K_fold-1)/K_fold * X.shape[0] - 1)), min(X.shape[1], int((K_fold-1)/K_fold * X.shape[0] - 1)), dtype = np.uint64)
        elif 'SPLS_K' not in kwargs or kwargs['SPLS_K'] is None:
            kwargs['SPLS_K'] = np.linspace(1, min(X.shape[1], X.shape[0]-1), min(X.shape[1], X.shape[0]-1), dtype = np.uint64)
        if ('SPLS_eta' not in kwargs or kwargs['SPLS_eta'] is None) and model_name == 'SPLS':
            kwargs['SPLS_eta'] = np.linspace(0, 1, 20, endpoint = False)
        elif 'SPLS_eta' not in kwargs or kwargs['SPLS_eta'] is None or model_name == 'PLS': # Normal PLS is simply SPLS with eta = 0
            kwargs['SPLS_eta'] = np.array([0])

        hyperparam_prod = list(product(kwargs['SPLS_K'], kwargs['SPLS_eta']))
        MSE_result = np.empty((len(kwargs['SPLS_K']) * len(kwargs['SPLS_eta']), K_fold*Nr)) * np.nan
        # Var = np.empty((len(kwargs['SPLS_K']) * len(kwargs['SPLS_eta']), X_unscaled.shape[1], K_fold*Nr)) * np.nan
        Var = np.empty((len(kwargs['SPLS_K']) * len(kwargs['SPLS_eta']), K_fold*Nr)) * np.nan # Var is n_hyperparams x n_folds because rm.SPLS_fitting() can return a different number of variables each time, so we do some processing to directly save the number of variables in the Var array

        with Parallel(n_jobs = -1) as PAR:
            for counter, (X_train, y_train, X_val, y_val) in enumerate(CVpartition(X_unscaled, y_unscaled, cv_type, K_fold, Nr, kwargs['cv_random_state'], group)):
                # Rescaling here to avoid validation dataset leakage
                if kwargs['scale_X'] and len(X.shape) == 2 and X.shape[1] > 0: # StandardScaler doesn't work with 3D arrays or with arrays that are 2D but empty in one dimension
                    scaler_x = StandardScaler(with_mean=True, with_std=True)
                    scaler_x.fit(X_train)
                    X_train_scale = scaler_x.transform(X_train)
                    X_val_scale = scaler_x.transform(X_val)
                else:
                    X_train_scale = X_train
                    X_val_scale = X_val
                if kwargs['scale_y'] and not kwargs['classification']:
                    scaler_y = StandardScaler(with_mean=True, with_std=True)
                    scaler_y.fit(y_train)
                    y_train_scale = scaler_y.transform(y_train)
                    y_val_scale = scaler_y.transform(y_val)
                else:
                    y_train_scale = y_train
                    y_val_scale = y_val
                temp = PAR(delayed(rm.SPLS_fitting)(X_train_scale, y_train_scale, X_val_scale, y_val_scale, this_prod[0], this_prod[1]) for this_prod in hyperparam_prod)
                _, temp_var_count, _, MSE_result[:, counter], _, _ = zip(*temp)
                Var[:, counter] = [np.sum(elem != 0) for elem in temp_var_count]

        MSE_mean = np.nanmean(MSE_result, axis = 1)
        # Min MSE value (first occurrence)
        ind = np.nanargmin(MSE_mean)
        if kwargs['robust_priority']:
            MSE_std = np.nanstd(MSE_result, axis = 1)
            MSE_min = MSE_mean[ind]
            MSE_bar = MSE_min + MSE_std[ind]
            Var_num = np.nansum(Var, axis = 1) # Here, Var is n_hyperparams x n_folds, and Var_num is n_hyperparams
            ind = np.nonzero( Var_num == np.nanmin(Var_num[MSE_mean < MSE_bar]) )[0][0] # Hyperparams with the lowest number of variables but still within one stdev of the best MSE

        # Hyperparameter setup
        K, eta = hyperparam_prod[ind]
        hyperparams = {'K': int(K), 'eta': float(eta)}
        # Fit the final model
        SPLS_model, SPLS_params, mse_train, mse_test, yhat_train, yhat_test = rm.SPLS_fitting(X, y, X_test, y_test, eta = eta, K = K)
        return(hyperparams, SPLS_model, SPLS_params, mse_train, mse_test, yhat_train, yhat_test, MSE_mean[ind])

    elif model_name == 'LCEN':
        kwargs['model_name'] = model_name # Sent to the joblib fun
        if 'degree' not in kwargs:
            kwargs['degree'] = [1, 2, 3]
        if 'label_name' not in kwargs: # Whether to auto-generate label names for the variables [x1, x2, ..., log(x1), ..., 1/x1, ..., x1*x2, etc.]
            kwargs['label_name'] = True # TODO: currently unused. Not sure whether I'll re-implement custom naming
        if 'trans_type' not in kwargs:
            kwargs['trans_type'] = 'all'
        if 'lag' not in kwargs:
            kwargs['lag'] = [0]
        if 'min_lag' not in kwargs:
            kwargs['min_lag'] = 0
        if 'LCEN_cutoff' not in kwargs:
            kwargs['LCEN_cutoff'] = 4e-2
        if 'LCEN_interaction' not in kwargs:
            kwargs['LCEN_interaction'] = True
        if 'LCEN_transform_y' not in kwargs:
            kwargs['LCEN_transform_y'] = False
        if 'LCEN_clip_min_classes_selected' not in kwargs:
            kwargs['LCEN_clip_min_classes_selected'] = 1
        elif kwargs['classification'] and kwargs['LCEN_clip_min_classes_selected'] > len(set(y)):
            raise ValueError(f'The "LCEN_clip_min_classes_selected" parameter should be <= the number of classes in the data. The value input for "LCEN_clip_min_classes_selected" was {kwargs["LCEN_clip_min_classes_selected"]}, but only {len(set(y))} classes were detected in the data.')
        kwargs['selection'] = None # for the _LCEN_joblib_fun called below
        kwargs['all_pos_X'] = np.all(X >= 0, axis = 0) & np.all(X_test >= 0, axis = 0) # Whether all entries in X are positive; used in _feature_trans() inside regression_models.py
        kwargs['all_pos_y'] = np.all(y >= 0, axis = 0) & np.all(y_test >= 0, axis = 0) # Whether all entries in y are positive; used in _feature_trans() inside regression_models.py
        if kwargs['LCEN_LASSO_coeff_file'] is None or not os_exists(kwargs['LCEN_LASSO_coeff_file']):
            # First run for variable selection using a L1_ratio of 1 (that is, only using an L1 penalty)
            hyperparam_prod = list(product(kwargs['degree'], [1], kwargs['alpha'], kwargs['lag']))
            if kwargs['verbosity_level'] >= 2: print(f'LCEN: beginning LASSO runs for feature selection. There are {len(hyperparam_prod)} hyperparameter combinations')
            with Parallel(n_jobs = -1) as PAR:
                if 'IC' in cv_type: # Information criterion
                    temp = PAR(delayed(_LCEN_joblib_fun)(X, y, X_test, y_test, eps, kwargs, prod_idx, this_prod) for prod_idx, this_prod in enumerate(hyperparam_prod))
                    temp = list(zip(*temp))[2] # To isolate the (AIC, AICc, BIC) tuple, which is the 3rd subentry of each entry in the original temp
                    temp = np.array(temp)
                    if cv_type == 'AICc':
                        IC_result = temp[:, 1]
                    elif cv_type == 'BIC':
                        IC_result = temp[:, 2]
                    else: # AIC
                        IC_result = temp[:, 0]
                    ind = np.argmin(IC_result)
                else: # Cross-validation
                    MSE_result = np.empty((len(kwargs['degree']) * len(kwargs['alpha']) * len(kwargs['lag']), K_fold*Nr)) * np.nan
                    for counter, (X_train, y_train, X_val, y_val) in enumerate(CVpartition(X, y, cv_type, K_fold, Nr, kwargs['cv_random_state'], group)):
                        temp = PAR(delayed(_LCEN_joblib_fun)(X_train, y_train, X_val, y_val, eps, kwargs,
                                prod_idx, this_prod, counter) for prod_idx, this_prod in enumerate(hyperparam_prod))
                        if not kwargs['classification']:
                            MSE_result[:, counter], _, _ = zip(*temp)
                        else:
                            loss_dicts, _, _ = zip(*temp)
                            MSE_result[:, counter] = [elem['MCC'] for elem in loss_dicts]
                    # Best hyperparameters for the preliminary run
                    MSE_mean = np.nanmean(MSE_result, axis = 1)
                    if not kwargs['classification']:
                        ind = np.nanargmin(MSE_mean) # Minimize the MSE
                    else:
                        ind = np.nanargmax(MSE_mean) # Maximize the MCC
            # Run to obtain the coefficients when LCEN is run with L1_ratio = 1
            degree, l1_ratio, LASSO_alpha, lag = hyperparam_prod[ind]
            LCEN_model, LCEN_params, _, _, _, _, label_names, _ = rm.LCEN_fitting(X, y, X_test, y_test, LASSO_alpha, l1_ratio, degree, lag, kwargs['min_lag'], kwargs['trans_type'],
                                            kwargs['LCEN_interaction'], kwargs['selection'], kwargs['LCEN_transform_y'], kwargs['all_pos_X'], kwargs['all_pos_y'],
                                            kwargs['scale_X'], kwargs['scale_y'], kwargs['model_random_state'], kwargs['classification'])
            if kwargs['LCEN_LASSO_coeff_file'] is not None:
                import json
                LASSO_coeff_file = {'degree': degree, 'l1_ratio': l1_ratio, 'LASSO_alpha': LASSO_alpha, 'lag': lag, 'LASSO_scaled_parameters': LCEN_params.tolist()}
                with open(kwargs['LCEN_LASSO_coeff_file'], 'w') as f:
                    json.dump(LASSO_coeff_file, f, indent = 4)
            elif kwargs['verbosity_level'] >= 2: print('NOTE: The "LCEN_LASSO_coeff_file" parameter was not set. If testing multiple LCEN_cutoff values, it is recommended to set it to a .json file (that does not exist) so that future runs on the same dataset are faster.')
        else:
            import json
            with open(kwargs['LCEN_LASSO_coeff_file']) as f:
                temp_dict = json.load(f)
            degree = temp_dict['degree']
            l1_ratio = temp_dict['l1_ratio']
            LASSO_alpha = temp_dict['LASSO_alpha']
            lag = temp_dict['lag']
            LCEN_params = np.array(temp_dict['LASSO_scaled_parameters'])
            print(f'LCEN: read LASSO coefficients from file {kwargs["LCEN_LASSO_coeff_file"]}')
        kwargs['selection'] = (np.abs(LCEN_params) >= kwargs['LCEN_cutoff'])&(np.abs(LCEN_params) != 0) # 1st clip step
        if kwargs['classification']: # Classification generates one set of features per class, but there is no way (to my knowledge) to run the EN step with a different set of features for each class
            kwargs['selection'] = kwargs['selection'].sum(axis = 0) >= kwargs['LCEN_clip_min_classes_selected']

        # Second run with a free L1_ratio but fixed degree and lag
        hyperparam_prod = list(product([degree], kwargs['l1_ratio'], kwargs['alpha'], [lag])) # Degree and lag have been fixed above
        if kwargs['verbosity_level'] >= 2: print(f'LCEN: beginning final runs. There are {len(hyperparam_prod)} hyperparameter combinations')
        with Parallel(n_jobs = -1) as PAR:
            if 'IC' in cv_type: # Information criterion
                temp = PAR(delayed(_LCEN_joblib_fun)(X, y, X_test, y_test, eps, kwargs, prod_idx, this_prod) for prod_idx, this_prod in enumerate(hyperparam_prod))
                temp = list(zip(*temp))[2] # To isolate the (AIC, AICc, BIC) tuple, which is the 3rd subentry of each entry in the original temp
                temp = np.array(temp)
                if cv_type == 'AICc':
                    IC_result = temp[:, 1]
                elif cv_type == 'BIC':
                    IC_result = temp[:, 2]
                else: # AIC
                    IC_result = temp[:, 0]
                ind = np.argmin(IC_result)
            else:
                MSE_result = np.empty((len(kwargs['alpha']) * len(kwargs['l1_ratio']), K_fold*Nr)) * np.nan
                Var = np.empty((len(kwargs['alpha']) * len(kwargs['l1_ratio']), K_fold*Nr)) * np.nan # Used when robust_priority == True
                for counter, (X_train, y_train, X_val, y_val) in enumerate(CVpartition(X, y, cv_type, K_fold, Nr, kwargs['cv_random_state'], group)):
                    temp = PAR(delayed(_LCEN_joblib_fun)(X_train, y_train, X_val, y_val, eps, kwargs, prod_idx,
                            this_prod, counter) for prod_idx, this_prod in enumerate(hyperparam_prod))
                    if not kwargs['classification']:
                        MSE_result[:, counter], Var[:, counter], _ = zip(*temp)
                    else:
                        loss_dicts, Var[:, counter], _ = zip(*temp)
                        MSE_result[:, counter] = [elem['MCC'] for elem in loss_dicts]
                # Best hyperparameters
                MSE_mean = np.nanmean(MSE_result, axis = 1)
                if not kwargs['classification']:
                    ind = np.nanargmin(MSE_mean) # Minimize the MSE
                else:
                    ind = np.nanargmax(MSE_mean) # Maximize the MCC
                if kwargs['robust_priority']:
                    MSE_std = np.nanstd(MSE_result, axis = 1)
                    MSE_min = MSE_mean[ind]
                    Var_num = np.nansum(Var, axis = 1) # Here, Var is n_hyperparams x n_folds, and Var_num is n_hyperparams
                    if not kwargs['classification']:
                        MSE_bar = MSE_min + MSE_std[ind]
                        ind = np.nonzero( Var_num == np.nanmin(Var_num[MSE_mean < MSE_bar]) )[0][0] # Hyperparams with the lowest number of variables but still within one stdev of the best MSE
                    else:
                        MSE_bar = MSE_min - MSE_std[ind] # Despite the name, MSE_min is more like MCC_max in this context
                        ind = np.nonzero( Var_num == np.nanmin(Var_num[MSE_mean > MSE_bar]) )[0][0] # Hyperparams with the lowest number of variables but still within one stdev of the best MCC

        # Hyperparameter setup
        degree, l1_ratio, alpha, lag = hyperparam_prod[ind]
        hyperparams = {'degree': degree, 'LASSO_alpha': LASSO_alpha, 'l1_ratio': l1_ratio, 'EN_alpha': alpha, 'lag': lag, 'cutoff': kwargs['LCEN_cutoff'], 'trans_type': kwargs['trans_type']}
        if kwargs['classification'] and len(set(y)) > 2: # The LCEN_clip_min_classes_selected is relevant only in multiclass classification
            hyperparams['LCEN_clip_min_classes_selected'] = kwargs['LCEN_clip_min_classes_selected']
        # Final run with the test set and best hyperparameters
        LCEN_model, LCEN_params, _, _, _, _, label_names, ICs = rm.LCEN_fitting(X, y, X_test, y_test, alpha, l1_ratio, degree, lag, kwargs['min_lag'], kwargs['trans_type'],
                                        kwargs['LCEN_interaction'], kwargs['selection'], kwargs['LCEN_transform_y'], kwargs['all_pos_X'], kwargs['all_pos_y'],
                                        kwargs['scale_X'], kwargs['scale_y'], kwargs['model_random_state'], kwargs['classification'])
        label_names = label_names[kwargs['selection']]
        # Removing the features that had small coefficients after the final model selection (2nd clip step)
        if kwargs['classification'] and len( set(y.squeeze()) ) == 2: # Binary classification generates only one set of features for the positive class, but the set of features for the negative class is just that of the positive class * -1
            LCEN_params = np.vstack((-LCEN_params, LCEN_params))
        final_selection = (np.abs(LCEN_params) >= kwargs['LCEN_cutoff'])&(np.abs(LCEN_params) != 0) # np.atleast_1d because if only one variable was passed to the final run, final_selection would be a bool instead of an array of bool w/o atleast_1d
        if LCEN_model and LCEN_model.coef_.ndim == 1:
            LCEN_model.coef_[~final_selection] = 0
        elif LCEN_model and len( set(y.squeeze()) ) == 2: # Binary classification
            LCEN_model.coef_[~np.atleast_2d(final_selection[1])] = 0 # final_selection is 2D, but the indexing makes it 1D, so np.atleast_2d is needed
        elif LCEN_model: # LCEN_model may be None
            LCEN_model.coef_[~np.atleast_2d(final_selection)] = 0 # final_selection is 1D in regression problems
        if not kwargs['classification']:
            LCEN_params = LCEN_params[final_selection].reshape(-1) # Reshape to avoid 0D arrays when only one variable is selected
            label_names = label_names[final_selection].reshape(-1)
        else:
            LCEN_params = np.array([LCEN_params[idx, final_selection[idx]] for idx in range(final_selection.shape[0])], dtype = object)
            label_names = [label_names[elem] for elem in final_selection]
        # Unscaling the model coefficients as per stackoverflow.com/questions/23642111/how-to-unscale-the-coefficients-from-an-lmer-model-fitted-with-a-scaled-respon
        if not kwargs['LCEN_transform_y'] and X.shape[1] > 0:
            X, X_test, _ = rm._feature_trans(X, X_test, degree, kwargs['LCEN_interaction'], kwargs['trans_type'], kwargs['all_pos_X'], kwargs['all_pos_y'])
        if lag > 0:
            X_temp = np.hstack([X[lag-1-idx : -idx-1, :] for idx in range(kwargs['min_lag'], lag)]) # The additional entries representing the previous times (t-1 to t-lag)
            y_temp = np.hstack([y[lag-1-idx:] if idx == 0 else y[lag-1-idx : -idx] for idx in range(kwargs['min_lag'], lag)])[lag:]
            X_test_temp = np.hstack([np.concatenate((X[X.shape[0]-idx-1:, :], X_test[: -idx-1, :])) for idx in range(kwargs['min_lag'], lag)]) # No need to remove entries from X_test or y_test because we can use the data from the final points of X or y to predict the initial points of X_test or y_test
            y_test_temp = np.hstack([y_test if idx == 0 else np.concatenate((y[(len(y)-idx):],  y_test[:-idx])) for idx in range(kwargs['min_lag'], lag)])
            X = np.hstack((X[lag:], X_temp, y_temp))
            y = y[lag:] # Shorterning y
            X_test = np.hstack((X_test, X_test_temp, y_test_temp))
        if kwargs['LCEN_transform_y']: # Feature transformation that includes the y features in the X and X_test variables
            X, X_test, _ = rm._feature_trans(X, X_test, degree, kwargs['LCEN_interaction'], kwargs['trans_type'], kwargs['all_pos_X'], kwargs['all_pos_y'])
        if not kwargs['classification']:
            LCEN_params_unscaled = np.zeros_like(LCEN_params)
            y_vars = np.array(['y' in label_names[idx] for idx in range(len(label_names))])
        else:
            LCEN_params_unscaled = np.empty_like(LCEN_params, dtype = object)
            for idx in range(LCEN_params.shape[0]):
                LCEN_params_unscaled[idx] = np.zeros_like(LCEN_params[idx])
            y_vars = [np.array(['y' in var for var in elem], dtype = bool) for elem in label_names]
        if len(label_names) and kwargs['scale_X'] and kwargs['scale_y'] and not kwargs['classification']: # Both X and y were scaled; regression [there should not be any classification with scaled y]
            LCEN_params_unscaled[~y_vars] = LCEN_params[~y_vars] * y.std() / X[:, kwargs['selection']][:, final_selection].std(axis=0)[~y_vars] # Unscaling the X variables # TODO: sometimes, when lag and min_lag > 0, an indexing problem occurs. Add [:, -len(kwargs['selection']):] before any X indexing to fix it
            LCEN_params_unscaled[y_vars] = LCEN_params[y_vars] * y.std() / X[:, kwargs['selection']][:, final_selection].std(axis=0)[y_vars]
        elif len(label_names) and kwargs['scale_X'] and not kwargs['classification']: # Only X was scaled; regression
            LCEN_params_unscaled[~y_vars] = LCEN_params[~y_vars] / X[:, kwargs['selection']][:, final_selection].std(axis=0)[~y_vars] # Unscaling the X variables # TODO: sometimes, when lag and min_lag > 0, an indexing problem occurs. Add [:, -len(kwargs['selection']):] before any X indexing to fix it
            LCEN_params_unscaled[y_vars] = LCEN_params[y_vars]
        elif len(label_names) and kwargs['scale_X'] and kwargs['classification']: # Only X was scaled; classification
            for this_class in range(len( set(y.squeeze()) )):
                LCEN_params_unscaled[this_class][~y_vars[this_class]] = LCEN_params[this_class][~y_vars[this_class]] / X[:, kwargs['selection']][:, final_selection[this_class]].std(axis=0)[~y_vars[this_class]] # Unscaling the X variables # TODO: sometimes, when lag and min_lag > 0, an indexing problem occurs. Add [:, -len(kwargs['selection']):] before any X indexing to fix it
                LCEN_params_unscaled[this_class][y_vars[this_class]] = LCEN_params[this_class][y_vars[this_class]]
        elif len(label_names) and kwargs['scale_y']: # Only y was scaled; regression
            if len(kwargs['all_pos_X']) and kwargs['verbosity_level']: print('WARNING: the unscaled coefficients are offset by a single constant. Fix TODO.') # TODO: if there is no X, this unscaling issue probably does not occur. len(kwargs['all_pos_X']) == 0 if there is no X
            LCEN_params_unscaled[~y_vars] = LCEN_params[~y_vars] * y.std()
            LCEN_params_unscaled[y_vars] = LCEN_params[y_vars] * y.std()
        elif len(label_names): # No scaling. If len(label_names) == False, no variables were selected
            LCEN_params_unscaled = LCEN_params
        # Obtaining the predictions again since coefficients may have been removed
        if not kwargs['classification']:
            yhat_train = np.dot(X[:, kwargs['selection']][:, final_selection], LCEN_params_unscaled) # TODO: sometimes, when lag and min_lag > 0, an indexing problem occurs. Add [:, -len(kwargs['selection']):] before any X indexing to fix it
            yhat_test = np.dot(X_test[:, kwargs['selection']][:, final_selection], LCEN_params_unscaled) # TODO: sometimes, when lag and min_lag > 0, an indexing problem occurs. Add [:, -len(kwargs['selection']):] before any X indexing to fix it
            intercept = (y.squeeze() - yhat_train).mean()
            if np.abs(intercept/y.mean()) >= kwargs['LCEN_cutoff']:
                yhat_train += intercept
                yhat_test += intercept
                label_names = np.concatenate((['intercept'], label_names))
                LCEN_params_unscaled = np.concatenate(([intercept], LCEN_params_unscaled))
            loss_train = MSE(y, yhat_train)
            loss_test = MSE(y_test, yhat_test)
        else:
            if kwargs['scale_X'] and X.shape[0] > 0 and X.shape[1] > 0:
                scaler_x = StandardScaler(with_mean=True, with_std=True)
                scaler_x.fit(X)
                X = scaler_x.transform(X)
                X_test = scaler_x.transform(X_test)
            if LCEN_model is not None:
                yhat_train = LCEN_model.predict_proba(X[:, kwargs['selection']])
                yhat_test = LCEN_model.predict_proba(X_test[:, kwargs['selection']])
            else:
                _, class_counts = np.unique(y, return_counts = True)
                most_common_class = np.argmax(class_counts)
                yhat_train = np.zeros( (X.shape[0], len(class_counts)) ); yhat_train[:, most_common_class] = 1
                yhat_test = np.zeros( (X_test.shape[0], len(class_counts)) ); yhat_test[:, most_common_class] = 1
            if len( set(y.squeeze()) ) == 2: # Binary classification
                pred_class_train = (yhat_train[:, 1] >= 0.5).astype(int) + y.min() # y.min() is a correction because classes may not start from 0 # TODO: for binary classification, allow thresholds
                pred_class_test = (yhat_test[:, 1] >= 0.5).astype(int) + y.min() # y.min() is a correction because classes may not start from 0
            else: # Multi-class classification
                pred_class_train = yhat_train.argmax(axis=1) + y.min() # y.min() is a correction because classes may not start from 0
                pred_class_test = yhat_test.argmax(axis=1) + y.min() # y.min() is a correction because classes may not start from 0
            loss_train = rm._classification_score(y, pred_class_train)
            loss_test = rm._classification_score(y_test, pred_class_test)
        # Returning the results
        if kwargs['verbosity_level'] >= 3 and not kwargs['classification']:
            print(f'{len(LCEN_params_unscaled) - int(label_names[0] == "intercept")} variables {"(and intercept)"*(label_names[0] == "intercept")} were selected' + ' '*20)
            print(f'The validation MSE was {MSE_mean[ind]:.3e}')
        elif kwargs['verbosity_level'] >= 3:
            print(f'The validation MCC was {MSE_mean[ind]:.3e}')
        # Transforming numpy arrays of features to list so that they can be saved within a .json file, and adding label names next to each feature
        if not kwargs['classification']:
            LCEN_params_unscaled = list(zip(label_names, LCEN_params_unscaled))
            if len(label_names):
                add_intercept = int(label_names[0] == 'intercept') # Potentially removing the intercept, as the unscaled parameters may have it but the scaled parameters never do
                LCEN_params = list(zip(label_names[add_intercept:], LCEN_params))
            else:
                LCEN_params = list(zip(label_names, LCEN_params)) # Just another empty list so we return something for LCEN_params
        elif label_names: # Classification and at least one feature was selected
            temp_vars = {}
            temp_vars_scaled = {}
            for this_class in range(len( set(y.squeeze()) )):
                temp_vars[f'Class {this_class}'] = list(zip(label_names[this_class], LCEN_params_unscaled[this_class]))
                temp_vars_scaled[f'Class {this_class}'] = list(zip(label_names[this_class], LCEN_params[this_class]))
            LCEN_params_unscaled = temp_vars
            LCEN_params = temp_vars_scaled
        else: # Classification and no features were selected
            LCEN_params_unscaled = []
            LCEN_params = []
        if 'IC' in cv_type:
            return(hyperparams, LCEN_model, LCEN_params_unscaled, LCEN_params, loss_train, loss_test, yhat_train, yhat_test, IC_result[ind])
        else:
            return(hyperparams, LCEN_model, LCEN_params_unscaled, LCEN_params, loss_train, loss_test, yhat_train, yhat_test, MSE_mean[ind])

    elif model_name == 'RF' or model_name == 'GBDT':
        if 'RF_n_estimators' not in kwargs:
            kwargs['RF_n_estimators'] = [10, 25, 50, 100, 200, 300]
        if 'RF_max_depth' not in kwargs:
            kwargs['RF_max_depth'] = [2, 3, 5, 10, 15, 20, 40]
        if 'RF_min_samples_leaf' not in kwargs:
            kwargs['RF_min_samples_leaf'] = [0.001, 0.01, 0.02, 0.05, 0.1]
        if 'RF_n_features' not in kwargs:
            kwargs['RF_n_features'] = [0.1, 0.25, 0.333, 0.5, 0.667, 0.75, 1.0]
        # learning_rate matters only for GBDT — set it to None to use RFs
        if model_name == 'RF':
            kwargs['learning_rate'] = [None]
        elif 'learning_rate' not in kwargs or len(kwargs['learning_rate']) == 0:
            kwargs['learning_rate'] = [0.01, 0.05, 0.1, 0.2]

        hyperparam_prod = list(product(kwargs['RF_n_estimators'], kwargs['RF_max_depth'], kwargs['RF_min_samples_leaf'], kwargs['RF_n_features'], kwargs['learning_rate']))
        MSE_result = np.empty( (len(kwargs['RF_n_estimators']) * len(kwargs['RF_max_depth']) * len(kwargs['RF_min_samples_leaf']) * len(kwargs['RF_n_features']) * len(kwargs['learning_rate']), K_fold*Nr) ) * np.nan
        """
        if kwargs['robust_priority']:
            # TODO: is this scoring system correct? It ignores the actual values of the parameters, caring only about their lengths and positions in the array.
            S = np.zeros((len(kwargs['max_depth']), len(kwargs['n_estimators']), len(kwargs['min_samples_leaf'])))
            for i in range(len(kwargs['max_depth'])):
                for j in range(len(kwargs['n_estimators'])):
                    for k in range(len(kwargs['min_samples_leaf'])):
                        S[i, j, k] = i/len(kwargs['max_depth']) - k/len(kwargs['min_samples_leaf'])
        """

        with Parallel(n_jobs = -1) as PAR:
            for counter, (X_train, y_train, X_val, y_val) in enumerate(CVpartition(X_unscaled, y_unscaled, cv_type, K_fold, Nr, kwargs['cv_random_state'], group)):
                if kwargs['verbosity_level'] >= 2: print(f'{model_name}: Beginning fold {counter+1}', end = '\r')
                # Rescaling here to avoid validation dataset leakage
                if kwargs['scale_X'] and len(X.shape) == 2 and X.shape[1] > 0: # StandardScaler doesn't work with 3D arrays or with arrays that are 2D but empty in one dimension
                    scaler_x = StandardScaler(with_mean=True, with_std=True)
                    scaler_x.fit(X_train)
                    X_train_scale = scaler_x.transform(X_train)
                    X_val_scale = scaler_x.transform(X_val)
                else:
                    X_train_scale = X_train
                    X_val_scale = X_val
                if kwargs['scale_y'] and not kwargs['classification']:
                    scaler_y = StandardScaler(with_mean=True, with_std=True)
                    scaler_y.fit(y_train)
                    y_train_scale = scaler_y.transform(y_train)
                    y_val_scale = scaler_y.transform(y_val)
                else:
                    y_train_scale = y_train
                    y_val_scale = y_val
                temp = PAR(delayed(rm.forest_fitting)(X_train_scale, y_train_scale, X_val_scale, y_val_scale, this_prod[0], this_prod[1], this_prod[2], this_prod[3],
                                    this_prod[4], classification = kwargs['classification']) for this_prod in hyperparam_prod)
                if not kwargs['classification']:
                    _, _, MSE_result[:, counter], _, _ = zip(*temp)
                else:
                    _, _, loss_dicts, _, _ = zip(*temp)
                    MSE_result[:, counter] = [elem['MCC'] for elem in loss_dicts]

        MSE_mean = np.nanmean(MSE_result, axis = 1)
        if not kwargs['classification']:
            ind = np.nanargmin(MSE_mean) # Minimize the MSE
        else:
            ind = np.nanargmax(MSE_mean) # Maximize the MCC
        """ # TODO: see above for the robust_priority changes
        if kwargs['robust_priority']:
            MSE_std = np.nanstd(MSE_result, axis = 1)
            MSE_min = MSE_mean[ind]
            MSE_bar = MSE_min + MSE_std[ind]
            ind = np.nonzero( S == np.nanmin(S[MSE_mean < MSE_bar]) )[0][0] # Hyperparams with the lowest number of variables but still within one stdev of the best MSE
        """

        # Hyperparameter setup
        n_estimators, max_depth, min_samples_leaf, n_features, learning_rate = hyperparam_prod[ind]
        if model_name == 'RF':
            hyperparams = {'n_estimators': n_estimators, 'max_depth': max_depth, 'min_samples_leaf': min_samples_leaf, 'n_features': n_features}
        else: # learning_rate matters only for GBDT
            hyperparams = {'n_estimators': n_estimators, 'max_depth': max_depth, 'min_samples_leaf': min_samples_leaf, 'n_features': n_features, 'learning_rate': learning_rate}
        # Fit the final model
        RF_model, loss_train, loss_test, yhat_train, yhat_test = rm.forest_fitting(X, y, X_test, y_test, n_estimators, max_depth, min_samples_leaf,
                                                            n_features, classification = kwargs['classification'])
        return(hyperparams, RF_model, loss_train, loss_test, yhat_train, yhat_test, MSE_mean[ind])

    elif model_name == 'SVM':
        if 'SVM_gamma' not in kwargs or kwargs['SVM_gamma'] is None:
            gd = 1/X.shape[1]
            kwargs['SVM_gamma'] = [gd/50, gd/10, gd/5, gd/2, gd, gd*2, gd*5, gd*10, gd*50]
        if 'SVM_C' not in kwargs:
            kwargs['SVM_C'] = [0.01, 0.1, 1, 10, 50, 100]
        if 'SVM_epsilon' not in kwargs:
            kwargs['SVM_epsilon'] = [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3]

        if not kwargs['classification']:
            hyperparam_prod = list(product(kwargs['SVM_gamma'], kwargs['SVM_C'], kwargs['SVM_epsilon']))
            MSE_result = np.empty( (len(kwargs['SVM_gamma']) * len(kwargs['SVM_C']) * len(kwargs['SVM_epsilon']), K_fold*Nr) ) * np.nan
        else: # SVM for classification does not use the epsilon hyperparameter
            hyperparam_prod = list(product(kwargs['SVM_gamma'], kwargs['SVM_C']))
            MSE_result = np.empty( (len(kwargs['SVM_gamma']) * len(kwargs['SVM_C']), K_fold*Nr) ) * np.nan
        """
        if kwargs['robust_priority']:
            # TODO: is this scoring system correct? It ignores the actual values of the parameters, caring only about their lengths and positions in the array.
            S = np.zeros((len(kwargs['C']), len(kwargs['gamma']), len(kwargs['epsilon'])))
            for i in range(len(kwargs['C'])):
                for j in range(len(kwargs['gamma'])):
                    for k in range(len(kwargs['epsilon'])):
                        S[i, j, k] = i/len(kwargs['C']) - j/len(kwargs['gamma']) - k/len(kwargs['epsilon'])
        """

        with Parallel(n_jobs = -1) as PAR:
            for counter, (X_train, y_train, X_val, y_val) in enumerate(CVpartition(X_unscaled, y_unscaled, cv_type, K_fold, Nr, kwargs['cv_random_state'], group)):
                if kwargs['verbosity_level'] >= 2: print(f'{model_name}: Beginning fold {counter+1}', end = '\r')
                # Rescaling here to avoid validation dataset leakage
                if kwargs['scale_X'] and len(X.shape) == 2 and X.shape[1] > 0: # StandardScaler doesn't work with 3D arrays or with arrays that are 2D but empty in one dimension
                    scaler_x = StandardScaler(with_mean=True, with_std=True)
                    scaler_x.fit(X_train)
                    X_train_scale = scaler_x.transform(X_train)
                    X_val_scale = scaler_x.transform(X_val)
                else:
                    X_train_scale = X_train
                    X_val_scale = X_val
                if kwargs['scale_y'] and not kwargs['classification']:
                    scaler_y = StandardScaler(with_mean=True, with_std=True)
                    scaler_y.fit(y_train)
                    y_train_scale = scaler_y.transform(y_train)
                    y_val_scale = scaler_y.transform(y_val)
                else:
                    y_train_scale = y_train
                    y_val_scale = y_val

                if not kwargs['classification']:
                    temp = PAR(delayed(rm.SVM_fitting)(X_train_scale, y_train_scale, X_val_scale, y_val_scale, this_prod[0], this_prod[1], this_prod[2]) for this_prod in hyperparam_prod)
                    _, _, MSE_result[:, counter], _, _ = zip(*temp)
                else: # SVM for classification does not use the epsilon hyperparameter
                    temp = PAR(delayed(rm.SVM_fitting)(X_train_scale, y_train_scale, X_val_scale, y_val_scale, this_prod[0], this_prod[1], classification = True) for this_prod in hyperparam_prod)
                    _, _, loss_dicts, _, _ = zip(*temp)
                    MSE_result[:, counter] = [elem['MCC'] for elem in loss_dicts]

        MSE_mean = np.nanmean(MSE_result, axis = 1)
        if not kwargs['classification']:
            ind = np.nanargmin(MSE_mean) # Minimize the MSE
        else:
            ind = np.nanargmax(MSE_mean) # Maximize the MCC
        """ # TODO: see above for the robust_priority changes
        if kwargs['robust_priority']:
            MSE_std = np.nanstd(MSE_result, axis = 1)
            MSE_min = MSE_mean[ind]
            MSE_bar = MSE_min + MSE_std[ind]
            ind = np.nonzero( S == np.nanmin(S[MSE_mean < MSE_bar]) )[0][0] # Hyperparams with the lowest number of variables but still within one stdev of the best MSE
        """

        # Hyperparameter setup
        if not kwargs['classification']:
            gamma, C, epsilon = hyperparam_prod[ind]
            hyperparams = {'gamma': gamma, 'C': C, 'epsilon': epsilon}
            SVM_model, loss_train, loss_test, yhat_train, yhat_test = rm.SVM_fitting(X, y, X_test, y_test, gamma, C, epsilon) # Fitting the final model
        else:
            gamma, C = hyperparam_prod[ind]
            hyperparams = {'gamma': gamma, 'C': C}
            SVM_model, loss_train, loss_test, yhat_train, yhat_test = rm.SVM_fitting(X, y, X_test, y_test, gamma, C, classification = True) # Fitting the final model
        return(hyperparams, SVM_model, loss_train, loss_test, yhat_train, yhat_test, MSE_mean[ind])

    elif model_name == 'AdaB':
        if 'RF_n_estimators' not in kwargs:
            kwargs['RF_n_estimators'] = [10, 25, 50, 100, 200, 300]
        if 'learning_rate' not in kwargs or len(kwargs['learning_rate']) == 0:
            kwargs['learning_rate'] = [0.01, 0.05, 0.1, 0.2]

        hyperparam_prod = list(product(kwargs['RF_n_estimators'], kwargs['learning_rate']))
        MSE_result = np.empty( (len(kwargs['RF_n_estimators']) * len(kwargs['learning_rate']), K_fold*Nr) ) * np.nan

        with Parallel(n_jobs = -1) as PAR:
            for counter, (X_train, y_train, X_val, y_val) in enumerate(CVpartition(X_unscaled, y_unscaled, cv_type, K_fold, Nr, kwargs['cv_random_state'], group)):
                # Rescaling here to avoid validation dataset leakage
                if kwargs['scale_X'] and len(X.shape) == 2 and X.shape[1] > 0: # StandardScaler doesn't work with 3D arrays or with arrays that are 2D but empty in one dimension
                    scaler_x = StandardScaler(with_mean=True, with_std=True)
                    scaler_x.fit(X_train)
                    X_train_scale = scaler_x.transform(X_train)
                    X_val_scale = scaler_x.transform(X_val)
                else:
                    X_train_scale = X_train
                    X_val_scale = X_val
                if kwargs['scale_y'] and not kwargs['classification']:
                    scaler_y = StandardScaler(with_mean=True, with_std=True)
                    scaler_y.fit(y_train)
                    y_train_scale = scaler_y.transform(y_train)
                    y_val_scale = scaler_y.transform(y_val)
                else:
                    y_train_scale = y_train
                    y_val_scale = y_val
                temp = PAR(delayed(rm.AdaBoost_fitting)(X_train_scale, y_train_scale, X_val_scale, y_val_scale, this_prod[0], this_prod[1],
                                    classification = kwargs['classification']) for this_prod in hyperparam_prod)
                if not kwargs['classification']:
                    _, _, MSE_result[:, counter], _, _ = zip(*temp)
                else:
                    _, _, loss_dicts, _, _ = zip(*temp)
                    MSE_result[:, counter] = [elem['MCC'] for elem in loss_dicts]

        MSE_mean = np.nanmean(MSE_result, axis = 1)
        if not kwargs['classification']:
            ind = np.nanargmin(MSE_mean) # Minimize the MSE
        else:
            ind = np.nanargmax(MSE_mean) # Maximize the MCC
        # Hyperparameter setup
        n_estimators, learning_rate = hyperparam_prod[ind]
        hyperparams = {'n_estimators': n_estimators, 'learning_rate': learning_rate}
        # Fit the final model
        AdaB_model, loss_train, loss_test, yhat_train, yhat_test = rm.AdaBoost_fitting(X, y, X_test, y_test, n_estimators, learning_rate, classification = kwargs['classification'])
        return(hyperparams, AdaB_model, loss_train, loss_test, yhat_train, yhat_test, MSE_mean[ind])

    elif model_name in {'MLP', 'RNN'}:
        kwargs['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Loss function
        if not kwargs['classification']:
            loss_function = torch.nn.functional.mse_loss
        elif 'diff_MCC_gamma' in kwargs and kwargs['diff_MCC_gamma'] is not None:
            loss_function = differentiable_MCC
        else:
            loss_function = torch.nn.CrossEntropyLoss(weight = torch.Tensor(kwargs['class_weight'])).to(kwargs['device'])
            kwargs['diff_MCC_gamma'] = None
        # Layer hyperparameters
        if kwargs['classification']:
            myshape_y = len( set(y.squeeze()) )
            min_y = y.min() # y.min() is a correction because classes may not start from 0
            y -= min_y
            y_unscaled -= min_y
            y_test -= min_y
        else:
            myshape_y = y.shape[1]
        if 'MLP_layers' not in kwargs or kwargs['MLP_layers'] is None:
            myshape_X = X.shape[1]
            kwargs['MLP_layers'] = [[(myshape_X, myshape_X*2), (myshape_X*2, myshape_y)], [(myshape_X, myshape_X), (myshape_X, myshape_y)], [(myshape_X, myshape_X//2), (myshape_X//2, myshape_y)], # One hidden layer
                                    [(myshape_X, myshape_X*2), (myshape_X*2, myshape_X*2), (myshape_X*2, myshape_y)], [(myshape_X, myshape_X*2), (myshape_X*2, myshape_X), (myshape_X, myshape_y)], # Two hidden layers
                                    [(myshape_X, myshape_X), (myshape_X, myshape_X), (myshape_X, myshape_y)], [(myshape_X, myshape_X), (myshape_X, myshape_X//2), (myshape_X//2, myshape_y)] ]
        if model_name == 'RNN':
            if 'RNN_layers' not in kwargs or kwargs['RNN_layers'] is None:
                kwargs['RNN_layers'] = X.shape[1]
            # Ensuring the MLP layers are compatible with the RNNs by making the MLP layers' first size equal to the output size of the last RNN
            last_lstm_size = kwargs['RNN_layers'] if isinstance(kwargs['RNN_layers'], int) else kwargs['RNN_layers'][-1] # For convenience, since kwargs['RNN_layers'] could be a single int or a list
            for cur_layer in kwargs['MLP_layers']:
                cur_layer[0] = (last_lstm_size, cur_layer[0][1])
        elif model_name == 'MLP':
            kwargs['RNN_layers'] = 0
        # Other model and training hyperparameters
        if 'activation' not in kwargs:
            kwargs['activation'] = ['relu']
        if 'batch_size' not in kwargs:
            kwargs['batch_size'] = 32
        if 'learning_rate' not in kwargs:
            kwargs['learning_rate'] = [1e-2, 5e-3]
        if 'weight_decay' not in kwargs:
            kwargs['weight_decay'] = 0
        if 'l1_penalty_factor' not in kwargs:
            kwargs['l1_penalty_factor'] = 0
        if 'n_epochs' not in kwargs:
            kwargs['n_epochs'] = 100
        # Scheduler hyperparameters
        if 'scheduler' not in kwargs:
            kwargs['scheduler'] = 'plateau'
        if 'scheduler_mode' not in kwargs:
            kwargs['scheduler_mode'] = 'min'
        if 'scheduler_factor' not in kwargs:
            kwargs['scheduler_factor'] = 0.5
        if 'scheduler_patience' not in kwargs:
            kwargs['scheduler_patience'] = 10
        if 'scheduler_min_lr' not in kwargs:
            kwargs['scheduler_min_lr'] = 1/16
        if 'scheduler_last_epoch' not in kwargs or kwargs['scheduler_last_epoch'] is None:
            kwargs['scheduler_last_epoch'] = kwargs['n_epochs'] - 30
        elif kwargs['scheduler_last_epoch'] < 0:
            kwargs['scheduler_last_epoch'] = kwargs['n_epochs'] + kwargs['scheduler_last_epoch'] # scheduler_last_epoch is < 0, so n_epochs + scheduler_last_epoch < n_epochs
        if 'scheduler_warmup' not in kwargs:
            kwargs['scheduler_warmup'] = 10
        if 'val_loss_file' not in kwargs or kwargs['val_loss_file'] is None:
            time_now = '-'.join([str(elem) for elem in localtime()[:6]]) # YYYY-MM-DD-hh-mm-ss
            kwargs['val_loss_file'] = f'ANN_val-loss_{time_now}.csv'
        if 'expand_hyperparameter_search' not in kwargs:
            kwargs['expand_hyperparameter_search'] = False

        def CV_model(X_unscaled, y_unscaled, loss_function, cv_type, K_fold, Nr, group, kwargs, hyperparam_list = None):
            """
            This function runs a cross-validation procedure for each combination of MLP / RNN hyperparameters.
            Results are saved in the kwargs['val_loss_file'] .csv file.
            """
            # File to record the validation losses
            if isinstance(kwargs['val_loss_file'], str):
                try:
                    if isinstance(loss_function, torch.nn.CrossEntropyLoss) or ('__name__' in dir(loss_function) and loss_function.__name__ == 'mse_loss'):
                        index_col = [0, 1]
                    elif '__name__' in dir(loss_function) and loss_function.__name__ == 'differentiable_MCC':
                        index_col = [0, 1, 2]
                    final_val_loss = pd.read_csv(kwargs['val_loss_file'], index_col = index_col)
                except FileNotFoundError:
                    if kwargs['verbosity_level'] >= 2: print('NOTE: no validation loss file was found, so the cross-validation will begin from the first set of hyperparameters.')
                    if '__name__' in dir(loss_function) and loss_function.__name__ == 'differentiable_MCC':
                        my_prod = list(product(kwargs['activation'], kwargs['learning_rate'], kwargs['diff_MCC_gamma']))
                    else:
                        my_prod = list(product(kwargs['activation'], kwargs['learning_rate']))
                    my_idx = pd.MultiIndex.from_tuples(my_prod)
                    final_val_loss = pd.DataFrame(np.nan, index = my_idx, columns = [str(elem) for elem in kwargs['MLP_layers']])
            else:
                final_val_loss = kwargs['val_loss_file'] # TODO: setup to pass a val_loss_file and a path

            # Train and validate
            if '__name__' in dir(loss_function) and loss_function.__name__ == 'differentiable_MCC' and hyperparam_list is None:
                hyperparam_list = list(product(kwargs['MLP_layers'], kwargs['learning_rate'], kwargs['activation'], kwargs['diff_MCC_gamma'])) # TODO: most likely, we can always pass kwargs['gamma'] anyways since there is a check for loss_function.__name__. If so, we can simplify this and other lines (e.g.: expand hyperparameter search)
            elif hyperparam_list is None:
                hyperparam_list = list(product(kwargs['MLP_layers'], kwargs['learning_rate'], kwargs['activation']))
            for cur_idx, cur_hp in enumerate(hyperparam_list): # cur_hp is (layers, lr, activation)
                # We added a new layer configuration to the hyperparameters
                if not str(cur_hp[0]) in list(final_val_loss.columns):
                    final_val_loss.insert(kwargs['MLP_layers'].index(cur_hp[0]), str(cur_hp[0]), np.nan) # layers.index to ensure consistent order
                # We added a new activation or learning rate to the hyperparameters
                if '__name__' in dir(loss_function) and loss_function.__name__ == 'differentiable_MCC' and not (cur_hp[2], cur_hp[1], cur_hp[3]) in final_val_loss.index.to_list():
                    final_val_loss.loc[(cur_hp[2], cur_hp[1], cur_hp[3]), :] = np.nan
                    final_val_loss = final_val_loss.sort_index(ascending = [True, False, False]) # Sorting the indices
                elif (isinstance(loss_function, torch.nn.CrossEntropyLoss) or ('__name__' in dir(loss_function) and loss_function.__name__ == 'mse_loss')) and not (cur_hp[2], cur_hp[1]) in final_val_loss.index.to_list(): # Need to manually check for the other loss functions else this gets called every time
                    final_val_loss.loc[(cur_hp[2], cur_hp[1]), :] = np.nan
                    final_val_loss = final_val_loss.sort_index(ascending = [True, False]) # Sorting the indices

                # Determining whether we have already ran this combination of hyperparameters
                if '__name__' in dir(loss_function) and loss_function.__name__ == 'differentiable_MCC':
                    not_already_done = np.isnan( final_val_loss.at[(cur_hp[2], cur_hp[1], cur_hp[3]), str(cur_hp[0])] )
                    first_print = f'{loss_function.__name__} (weight = {kwargs["class_weight"]}): Beginning hyperparameters {cur_idx+1:2}/{len(hyperparam_list)}; layers = {cur_hp[0]}, LR = {cur_hp[1]}, act = {cur_hp[2]}, gamma = {cur_hp[3]}'
                else:
                    not_already_done = np.isnan( final_val_loss.at[(cur_hp[2], cur_hp[1]), str(cur_hp[0])] )
                    loss_name = f'CE (weight = {kwargs["class_weight"]})' if isinstance(loss_function, torch.nn.CrossEntropyLoss) else 'MSE'
                    first_print = f'{loss_name}: Beginning hyperparameters {cur_idx+1:2}/{len(hyperparam_list)}; layers = {cur_hp[0]}, LR = {cur_hp[1]}, act = {cur_hp[2]}'

                # Run CV only if we do not have validation losses for this set of parameters
                if not_already_done:
                    if kwargs['verbosity_level'] >= 2: print(first_print + ' '*7, end = '\r')
                    scheduler_min_lr = kwargs['scheduler_min_lr'] * cur_hp[1]
                    temp_val_loss = 0
                    for counter, (X_train, y_train, X_val, y_val) in enumerate(CVpartition(X_unscaled, y_unscaled, cv_type, K_fold, Nr, kwargs['cv_random_state'], group)):
                        if kwargs['verbosity_level'] >= 3: print(f'Current fold: {counter+1}' + ' '*18, end = '\r')
                        # Rescaling to avoid validation dataset leakage
                        if kwargs['scale_X'] and len(X.shape) == 2 and X.shape[1] > 0: # StandardScaler doesn't work with 3D arrays
                            scaler_x_train = StandardScaler(with_mean=True, with_std=True)
                            scaler_x_train.fit(X_train)
                            X_train_scale = torch.Tensor(scaler_x_train.transform(X_train))
                            X_val_scale = torch.Tensor(scaler_x_train.transform(X_val))
                        else:
                            X_train_scale = torch.Tensor(X_train)
                            X_val_scale = torch.Tensor(X_val)
                        if kwargs['scale_y'] and not kwargs['classification']:
                            scaler_y_train = StandardScaler(with_mean=True, with_std=True)
                            scaler_y_train.fit(y_train)
                            y_train_scale = torch.Tensor(scaler_y_train.transform(y_train))
                            y_val_scale = torch.Tensor(scaler_y_train.transform(y_val))
                        elif kwargs['classification']:
                            y_train_scale = torch.LongTensor(y_train)
                            y_val_scale = torch.LongTensor(y_val)
                        else:
                            y_train_scale = torch.Tensor(y_train)
                            y_val_scale = torch.Tensor(y_val)
                        # Creating the Datasets / DataLoaders + run setup
                        train_dataset_fold = MyDataset(X_train_scale, y_train_scale)
                        train_loader_fold = DataLoader(train_dataset_fold, kwargs['batch_size'], shuffle = True)
                        val_dataset_fold = MyDataset(X_val_scale, y_val_scale)
                        val_loader_fold = DataLoader(val_dataset_fold, kwargs['batch_size'], shuffle = True)
                        best_metric_fold = 1000 * np.maximum(1, y_train_scale.max().item()) # In regression, this should work but has not been fully tested. In classification, this gets compared to -F1, so any positive value is fine
                        epochs_with_improvement = []
                        epochs_with_improvement_print = []
                        # Declaring the model and optimizer
                        model = my_ANN(cur_hp[0], cur_hp[2], X_train_scale.shape[-1], kwargs['RNN_layers'], kwargs['device']).to(kwargs['device'])
                        optimizer = torch.optim.AdamW(model.parameters(), lr = cur_hp[1], weight_decay = kwargs['weight_decay'])
                        if kwargs['scheduler'].casefold() == 'plateau':
                            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, kwargs['scheduler_mode'], kwargs['scheduler_factor'], kwargs['scheduler_patience'], min_lr = scheduler_min_lr)
                        elif kwargs['scheduler'].casefold() == 'cosine':
                            scheduler = CosineScheduler(kwargs['scheduler_last_epoch'], cur_hp[1], warmup_steps = kwargs['scheduler_warmup'], final_lr = scheduler_min_lr)
                        elif kwargs['scheduler'].casefold() == 'lambda':
                            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, kwargs['scheduler_factor'])
                        elif kwargs['scheduler'].casefold() == 'step':
                            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, kwargs['scheduler_patience'], kwargs['scheduler_factor'])
                        elif kwargs['scheduler'].casefold() == 'multistep':
                            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, kwargs['scheduler_patience'], kwargs['scheduler_factor'])
                        elif kwargs['scheduler'].casefold() == 'exponential':
                            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, kwargs['scheduler_factor'])
                        for epoch in range(kwargs['n_epochs']):
                            if kwargs['verbosity_level'] >= 4 and not kwargs['classification']:
                                print(f'Fold {counter+1}; epoch {epoch+1:2}/{kwargs["n_epochs"]}; Best val loss = {best_metric_fold:5.2f}; Imp: {epochs_with_improvement_print}' + ' '*5, end = '\r')
                            elif kwargs['verbosity_level'] >= 4:
                                print(f'Fold {counter+1}; epoch {epoch+1:2}/{kwargs["n_epochs"]}; Best val F1 = {-best_metric_fold:6.4f}; Imp: {epochs_with_improvement_print}' + ' '*5, end = '\r')
                            if '__name__' in dir(loss_function) and loss_function.__name__ == 'differentiable_MCC':
                                train_loss, _ = loop_model(model, optimizer, train_loader_fold, loss_function, epoch, kwargs['batch_size'], myshape_y, classification = kwargs['classification'], gamma = cur_hp[3], loss_weight = kwargs['class_weight'], l1_penalty_factor = kwargs['l1_penalty_factor'])
                                val_loss, _ = loop_model(model, optimizer, val_loader_fold, loss_function, epoch, kwargs['batch_size'], myshape_y, evaluation = True, classification = kwargs['classification'], gamma = cur_hp[3], loss_weight = kwargs['class_weight'], l1_penalty_factor = kwargs['l1_penalty_factor'])
                            else:
                                train_loss, _ = loop_model(model, optimizer, train_loader_fold, loss_function, epoch, kwargs['batch_size'], myshape_y, classification = kwargs['classification'], l1_penalty_factor = kwargs['l1_penalty_factor'])
                                val_loss, _ = loop_model(model, optimizer, val_loader_fold, loss_function, epoch, kwargs['batch_size'], myshape_y, evaluation = True, classification = kwargs['classification'], l1_penalty_factor = kwargs['l1_penalty_factor'])
                            # Recording the improvement if current val loss < best val loss
                            if val_loss < best_metric_fold:
                                best_metric_fold = val_loss
                                epochs_with_improvement.append(epoch+1)
                                if not kwargs['classification']:
                                    epochs_with_improvement_print.append(f'{epoch+1}:{best_metric_fold:.1e}')
                                else:
                                    epochs_with_improvement_print.append(f'{epoch+1}:{-best_metric_fold:.3f}')
                            if 'scheduler' in locals() and scheduler.__module__ == 'torch.optim.lr_scheduler': # Pytorch built-in scheduler
                                scheduler.step(val_loss)
                            elif 'scheduler' in locals():
                                for param_group in optimizer.param_groups:
                                    param_group['lr'] = scheduler(epoch)
                        # Recording the validation loss for this fold
                        if kwargs['verbosity_level'] >= 3 and not kwargs['classification']:
                            print(f'Fold {counter+1} done; Best val loss = {best_metric_fold:5.2f}; Improvement = {epochs_with_improvement}' + ' '*18)
                        elif kwargs['verbosity_level'] >= 3:
                            print(f'Fold {counter+1} done; Best val F1 = {-best_metric_fold:6.4f}; Improvement = {epochs_with_improvement}' + ' '*18)
                        temp_val_loss += best_metric_fold
                    # Saving the average validation loss after CV
                    if '__name__' in dir(loss_function) and loss_function.__name__ == 'differentiable_MCC':
                        final_val_loss.at[(cur_hp[2], cur_hp[1], cur_hp[3]), str(cur_hp[0])] = temp_val_loss / (counter+1) # counter+1 because the correct value may be K_fold (for simple Kfold) or K_fold*Nr (for repeated K_fold)
                    else:
                        final_val_loss.at[(cur_hp[2], cur_hp[1]), str(cur_hp[0])] = temp_val_loss / (counter+1) # counter+1 because the correct value may be K_fold (for simple Kfold) or K_fold*Nr (for repeated K_fold)
                    final_val_loss.to_csv(kwargs['val_loss_file']) # TODO: setup to pass a val_loss_file and a path
            return final_val_loss
        final_val_loss = CV_model(X_unscaled, y_unscaled, loss_function, cv_type, K_fold, Nr, group, kwargs)
        # Finding the best hyperparameters
        best_idx, best_LR, best_neurons, best_act, best_act_loc, best_diff_MCC_gamma = _get_best_hyperparameters(final_val_loss)
        # Checking whether the best hyperparameters are in the extremes of what was cross-validated
        extreme_LR = best_idx[0] in {best_act_loc.start, best_act_loc.stop-1} # Whether the best LR was either the highest or the lowest value checked for the best activation function
        layer_mask = np.array([len(elem) for elem in kwargs['MLP_layers']]) == len(best_neurons) # Check only MLP configurations with the same number of layers
        layers_for_extreme = sorted([elem for idx, elem in enumerate(kwargs['MLP_layers']) if layer_mask[idx]])
        if len(layers_for_extreme) >= 2:
            extreme_neuron = best_neurons in [layers_for_extreme[0], layers_for_extreme[-1]] # Whether the best neuron was either the largest or the smallest combination checked # TODO: MLP configurations with more than 1 hidden layer will benefit from a more thorough analysis of extremity
        else:
            extreme_neuron = False
        while (extreme_LR or extreme_neuron) and kwargs['expand_hyperparameter_search']:
            print(f'Expanding the hyperparameters using the "{kwargs["expand_hyperparameter_search"]}" rule')
            hyperparam_list = list(product(kwargs['MLP_layers'], kwargs['learning_rate'], kwargs['activation'])) # Used when kwargs['expand_hyperparameter_search'] == 'single'
            # New LR value
            if best_idx[0] == best_act_loc.start: # The best LR was the highest value tested
                new_LR = best_LR * np.sqrt(10)
                kwargs['learning_rate'].append(new_LR)
            elif best_idx[0] == best_act_loc.stop-1: # The best LR was the lowest value tested
                new_LR = best_LR / np.sqrt(10)
                kwargs['learning_rate'].append(new_LR)
            # New MLP layer configuration
            new_neurons = best_neurons[:] # [:] makes a copy
            if best_neurons == layers_for_extreme[0] and best_neurons[0][1] != 1: # The best layer configuration was the smallest configuration tested. != 1 to avoid infinite loop
                if best_neurons[0][1] > X.shape[1]:
                    new_neurons[0] = (new_neurons[0][0], new_neurons[0][1] - X.shape[1])
                    new_neurons[1] = (new_neurons[1][0] - X.shape[1], new_neurons[0][0])
                else: # The layer is <= X.shape[1], so we'll simply divide it by 2
                    new_neurons[0] = (new_neurons[0][0], new_neurons[0][1] // 2)
                    new_neurons[1] = (new_neurons[1][0] // 2, new_neurons[0][0])
                kwargs['MLP_layers'].append(new_neurons)
                if len(best_neurons) > 2: # 2 (or more) hidden layers TODO
                    pass
            elif best_neurons == layers_for_extreme[-1]: # The best layer configuration was the largest configuration tested
                new_neurons[0] = (new_neurons[0][0], new_neurons[0][1] + X.shape[1])
                new_neurons[1] = (new_neurons[1][0] + X.shape[1], new_neurons[0][0])
                kwargs['MLP_layers'].append(new_neurons)
                if len(best_neurons) > 2: # 2 (or more) hidden layers TODO
                    pass
            if kwargs['expand_hyperparameter_search'].casefold() == 'grid':
                if best_act not in kwargs['activation']:
                    print(f'WARNING: the best activation function ({best_act}) is not among the input activations, which means hyperparameters using it will not be tested\n' +
                          f'\tInclude it in the SPA call (as activation = ["{best_act}", "other", "activations"]) to ensure its hyperparameters, too, will be tested.')
                final_val_loss = CV_model(X_unscaled, y_unscaled, loss_function, cv_type, K_fold, Nr, group, kwargs)
            elif kwargs['expand_hyperparameter_search'].casefold() == 'single':
                if 'new_LR' in locals() and 'new_neurons' in locals() and (new_neurons, new_LR, best_act) not in hyperparam_list: # Adding both a new LR and a new layer configuration
                    if '__name__' in dir(loss_function) and loss_function.__name__ == 'differentiable_MCC':
                        hyperparam_list.append((new_neurons, new_LR, best_act, best_diff_MCC_gamma))
                    else: # MSE and Cross-entropy
                        hyperparam_list.append((new_neurons, new_LR, best_act))
                if 'new_LR' in locals() and (best_neurons, new_LR, best_act) not in hyperparam_list: # Adding a new LR with the other best hyperparameters
                    if '__name__' in dir(loss_function) and loss_function.__name__ == 'differentiable_MCC':
                        hyperparam_list.append((best_neurons, new_LR, best_act, best_diff_MCC_gamma))
                    else: # MSE and Cross-entropy
                        hyperparam_list.append((best_neurons, new_LR, best_act))
                    del new_LR # To avoid an infinite loop, as the if statement check for the presence of this variable in locals()
                if 'new_neurons' in locals() and (new_neurons, best_LR, best_act) not in hyperparam_list: # Adding a new layer configuration with the other best hyperparameters
                    if '__name__' in dir(loss_function) and loss_function.__name__ == 'differentiable_MCC':
                        hyperparam_list.append((best_neurons, new_LR, best_act, best_diff_MCC_gamma))
                    else: # MSE and Cross-entropy
                        hyperparam_list.append((new_neurons, best_LR, best_act))
                    del new_neurons # To avoid an infinite loop, as the if statement check for the presence of this variable in locals()
                final_val_loss = CV_model(X_unscaled, y_unscaled, loss_function, cv_type, K_fold, Nr, group, kwargs, hyperparam_list)
            # Finding the best hyperparameters
            best_idx, best_LR, best_neurons, best_act, best_act_loc, best_diff_MCC_gamma = _get_best_hyperparameters(final_val_loss)
            # Checking whether the best hyperparameters are in the extremes of what was cross-validated
            extreme_LR = best_idx[0] in {best_act_loc.start, best_act_loc.stop-1} # Whether the best LR was either the highest or the lowest value checked for the best activation function
            layer_mask = np.array([len(elem) for elem in kwargs['MLP_layers']]) == len(best_neurons) # Check only MLP configurations with the same number of layers
            layers_for_extreme = np.sort(np.array(kwargs['MLP_layers'], dtype = object)[layer_mask])
            extreme_neuron = (best_neurons in [layers_for_extreme[0], layers_for_extreme[-1]] and best_neurons[0][1] != 1) # Whether the best neuron was either the largest or the smallest combination checked # TODO: MLP configurations with more than 1 hidden layer will benefit from a more thorough analysis of extremity
        # Creating LongTensors if running a classification task
        if not kwargs['classification']:
            y = torch.Tensor(y)
            y_test = torch.Tensor(y_test)
        else:
            y = torch.LongTensor(y)
            y_test = torch.LongTensor(y_test)
        # Final model training
        train_dataset = MyDataset(torch.Tensor(X), y)
        train_loader = DataLoader(train_dataset, kwargs['batch_size'], shuffle = True)
        # Re-declaring the model
        model = my_ANN(best_neurons, best_act, X.shape[-1], kwargs['RNN_layers'], kwargs['device']).to(kwargs['device'])
        optimizer = torch.optim.AdamW(model.parameters(), lr = best_LR, weight_decay = kwargs['weight_decay'])
        if kwargs['scheduler'].casefold() in {'plateau', 'cosine'}: # The plateau scheduler cannot work here because there is no validation loss, so it gets switched for the cosine scheduler
            scheduler = CosineScheduler(kwargs['scheduler_last_epoch'], base_lr = best_LR, warmup_steps = kwargs['scheduler_warmup'], final_lr = best_LR * kwargs['scheduler_min_lr'])
        # Retrain
        if kwargs['verbosity_level'] >= 2: print(f'Beginning final ANN training - MLP size = {best_neurons}; LR = {best_LR}; activation = {best_act}; batch size = {kwargs["batch_size"]}; weight_decay = {kwargs["weight_decay"]}')
        for epoch in range(kwargs['n_epochs']):
            loop_model(model, optimizer, train_loader, loss_function, epoch, kwargs['batch_size'], myshape_y, classification = kwargs['classification'], gamma = best_diff_MCC_gamma, loss_weight = kwargs['class_weight'], l1_penalty_factor = kwargs['l1_penalty_factor'])
            if 'scheduler' in locals() and scheduler.__module__ == 'torch.optim.lr_scheduler': # Pytorch built-in scheduler
                scheduler.step() # TODO: we do not really have a val_loss here. Need to check how the other built-in Schedulers behave
            elif 'scheduler' in locals():
                for param_group in optimizer.param_groups:
                    param_group['lr'] = scheduler(epoch)
        # Final evaluation - Train set
        train_dataset = MyDataset(torch.Tensor(X), y)
        train_loader = DataLoader(train_dataset, kwargs['batch_size'], shuffle = False) # Not shuffling because this is just for evaluation
        model.eval()
        train_pred = torch.empty((len(train_loader.dataset), myshape_y))
        if not kwargs['classification']:
            train_y = torch.empty(len(train_loader.dataset))
        else:
            train_y = torch.empty(len(train_loader.dataset), dtype = torch.long)
            if not ('__name__' in dir(loss_function) and loss_function.__name__ == 'differentiable_MCC'):
                loss_function = loss_function.cpu()
        for idx, data in enumerate(train_loader):
            X_finaleval, y_finaleval = data # _finaleval to avoid overwriting the original X and y
            X_finaleval = X_finaleval.cuda()
            pred = model(X_finaleval).cpu().detach()
            train_pred[idx*kwargs['batch_size']:(idx*kwargs['batch_size'])+len(pred), :] = pred
            train_y[idx*kwargs['batch_size']:(idx*kwargs['batch_size'])+len(y_finaleval)] = y_finaleval.squeeze()
        train_loss = loss_function(train_pred.squeeze(), train_y).item()
        # Final evaluation - Test set
        test_dataset = MyDataset(torch.Tensor(X_test), y_test)
        test_loader = DataLoader(test_dataset, kwargs['batch_size'], shuffle = False) # Not shuffling because this is just for evaluation
        test_pred = torch.empty((len(test_loader.dataset), myshape_y))
        if not kwargs['classification']:
            test_y = torch.empty(len(test_loader.dataset))
        else:
            test_y = torch.empty(len(test_loader.dataset), dtype = torch.long)
        for idx, data in enumerate(test_loader):
            X_finaleval, y_finaleval = data # _finaleval to avoid overwriting the original X and y
            X_finaleval = X_finaleval.cuda()
            pred = model(X_finaleval).cpu().detach()
            test_pred[idx*kwargs['batch_size']:(idx*kwargs['batch_size'])+len(pred), :] = pred
            test_y[idx*kwargs['batch_size']:(idx*kwargs['batch_size'])+len(y_finaleval)] = y_finaleval.squeeze()
        test_loss = loss_function(test_pred.squeeze(), test_y).item()
        best_hyperparameters = {'RNN size': str(kwargs['RNN_layers']), 'MLP size': str(best_neurons), 'LR': best_LR, 'activation': best_act, 'batch_size': kwargs['batch_size'], 'n_epochs': kwargs['n_epochs'], 'weight_decay': kwargs['weight_decay'], 'scheduler': kwargs['scheduler']}
        if kwargs['scheduler'].casefold() in {'plateau', 'lambda', 'step', 'multistep', 'exponential'}:
            best_hyperparameters['scheduler_factor'] = kwargs['scheduler_factor']
        if kwargs['scheduler'].casefold() in {'plateau', 'step', 'multistep'}:
            best_hyperparameters['scheduler_patience'] = kwargs['scheduler_patience']
        if kwargs['scheduler'].casefold() == 'cosine':
            best_hyperparameters['scheduler_warmup'] = kwargs['scheduler_warmup']
            best_hyperparameters['scheduler_last_epoch'] = kwargs['scheduler_last_epoch']
            best_hyperparameters['scheduler_min_lr'] = kwargs['scheduler_min_lr']
        if kwargs['classification']:
            temp_train_loss = train_loss
            train_pred = softmax(np.array(train_pred, dtype = float), axis = 1)
            train_pred[train_pred < 1e-20] = 0 # For convenience to eliminate very small numbers
            pred_class_train = train_pred.argmax(axis = 1) + min_y # min_y is a correction because classes may not start from 0
            train_loss = rm._classification_score(y + min_y, pred_class_train)
            temp_test_loss = test_loss
            test_pred = softmax(np.array(test_pred, dtype = float), axis = 1)
            test_pred[test_pred < 1e-20] = 0 # For convenience to eliminate very small numbers
            pred_class_test = test_pred.argmax(axis = 1) + min_y # min_y is a correction because classes may not start from 0
            test_loss = rm._classification_score(y_test + min_y, pred_class_test)
            if isinstance(loss_function, torch.nn.CrossEntropyLoss):
                train_loss['Cross Entropy loss'] = temp_train_loss
                test_loss['Cross Entropy loss'] = temp_test_loss
            elif '__name__' in dir(loss_function) and loss_function.__name__ == 'differentiable_MCC':
                best_hyperparameters['diff_MCC_gamma'] = best_diff_MCC_gamma
        return model, final_val_loss, train_loss, test_loss, train_pred, test_pred, best_hyperparameters

@ignore_warnings()
def _LCEN_joblib_fun(X_train, y_train, X_val, y_val, eps, kwargs, prod_idx, this_prod, counter = -1):
    """
    A helper function to parallelize LCEN. Shouldn't be called by the user
    """
    degree, l1_ratio, alpha, lag = this_prod
    if (prod_idx == 0 or not (prod_idx+1)%100) and counter >= 0 and kwargs['verbosity_level'] >= 2: # CV
        print(f'Beginning run {prod_idx+1:4} of fold {counter+1:3}', end = '\r')
    elif (prod_idx == 0 or not (prod_idx+1)%100) and kwargs['verbosity_level'] >= 2: # IC -- no folds
        print(f'Beginning run {prod_idx+1:4}', end = '\r')
    _, variable, _, mse, _, _, _, ICs = rm.LCEN_fitting(X_train, y_train, X_val, y_val, alpha, l1_ratio, degree, lag, kwargs['min_lag'], kwargs['trans_type'],
                            kwargs['LCEN_interaction'], kwargs['selection'], kwargs['LCEN_transform_y'], kwargs['all_pos_X'], kwargs['all_pos_y'],
                            kwargs['scale_X'], kwargs['scale_y'], kwargs['model_random_state'], kwargs['classification'])
    return mse, np.sum(variable.flatten() != 0), ICs

class MyDataset(Dataset):
    def __init__(self, Xdata, ydata):
        self.Xdata = Xdata
        self.ydata = ydata

    def __len__(self):
        return len(self.Xdata)

    def __getitem__(self, idx):
        return self.Xdata[idx], self.ydata[idx]

class CosineScheduler: # For MLPs and RNNs. Code obtained from https://d2l.ai/chapter_optimization/lr-scheduler.html
    def __init__(self, max_update, base_lr = 0.01, final_lr = 0, warmup_steps = 0, warmup_begin_lr = 0):
        self.base_lr_orig = base_lr
        self.max_update = max_update
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.warmup_begin_lr = warmup_begin_lr
        self.max_steps = self.max_update - self.warmup_steps

    def get_warmup_lr(self, epoch):
        increase = (self.base_lr_orig - self.warmup_begin_lr) * float(epoch) / float(self.warmup_steps)
        return self.warmup_begin_lr + increase

    def __call__(self, epoch):
        if epoch < self.warmup_steps:
            return self.get_warmup_lr(epoch)
        if epoch <= self.max_update:
            self.base_lr = self.final_lr + (
                self.base_lr_orig - self.final_lr) * (1 + np.cos(
                np.pi * (epoch - self.warmup_steps) / self.max_steps)) / 2
        return self.base_lr

# A helper function that is called every epoch of training or validation for MLPs and RNNs
def loop_model(model, optimizer, loader, loss_function, epoch, batch_size, shape_y, evaluation = False, classification = False, gamma = 1, loss_weight = 1, l1_penalty_factor = 0):
    if evaluation:
        model.eval()
        val_pred = torch.empty((len(loader.dataset), shape_y))
        val_y = torch.empty((len(loader.dataset)), dtype = torch.long)
    else:
        model.train()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    total_loss = 0
    for idx, data in enumerate(loader):
        X, y = data
        X = X.to(device)
        if y.ndim > 1:
            y = y.squeeze(1).to(device)
        else:
            y = y.to(device)
        pred = model(X, classification)
        if 'total_pred_y' not in locals():
            total_pred_y = torch.empty((len(loader.dataset), pred.shape[1]))
        total_pred_y[idx*batch_size:(idx*batch_size)+len(pred), :] = pred.cpu().detach()
        # Group L1 penalty
        l1_penalty = 0
        if l1_penalty_factor:
            params = model.named_parameters()
            for parameter_name, parameter in params:
                if 'weight' in parameter_name:
                    l1_penalty += torch.sum( np.sqrt(parameter.shape[0]) * torch.sqrt(torch.sum(parameter**2, axis=0)) )
        if '__name__' in dir(loss_function) and loss_function.__name__ == 'differentiable_MCC':
            loss = loss_function(pred.squeeze(1), y, loss_weight, gamma) + l1_penalty_factor * l1_penalty
        else:
            loss = loss_function(pred.squeeze(1), y) + l1_penalty_factor * l1_penalty
        total_loss += y.shape[0]*loss.item()/loader.dataset.ydata.shape[0]
        # Backpropagation
        if not evaluation:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        elif classification:
            val_pred[idx*batch_size:(idx*batch_size)+len(pred), :] = pred.cpu().detach()
            val_y[idx*batch_size:(idx*batch_size)+len(y)] = y
    if evaluation and classification:
        val_pred_CM = val_pred.argmax(axis=1)
        CM = confusion_matrix(val_y, val_pred_CM) # Confusion matrix to make F1 calcs easier
        if CM.shape[0] == 2: # Binary classification
            if CM[1,1]+CM[1,0] and CM[1,1]+CM[0,1]: # Avoids dividing by 0
                rec = CM[1,1]/(CM[1,1]+CM[1,0])
                pre = CM[1,1]/(CM[1,1]+CM[0,1])
            else:
                rec, pre = 0, 0
        else: # Multiclass classification
            rec = np.diag(CM)/CM.sum(axis=1)
            rec[np.isnan(rec)] = 0
            pre = np.diag(CM)/CM.sum(axis=0)
            pre[np.isnan(pre)] = 0
        if np.any(rec) and np.any(pre): # Avoids dividing by 0 when calculating F1
            total_loss = -2/(1/rec + 1/pre) # Calling it "total_loss" for convienience, but this is just the F1 score. A differentiable loss function is used above for backpropagation
        else:
            total_loss = 0
        if CM.shape[0] > 2 and not isinstance(total_loss, (int, float)): # Multiclass classification
            total_loss = total_loss.mean()
    return total_loss, total_pred_y

# MLP or LSTM+MLP model
class my_ANN(torch.nn.Module):
    def __init__(self, layers, activ_fun = 'relu', lstm_input_size = 0, lstm_hidden_size = 0, device = 'cuda'):
        super(my_ANN, self).__init__()
        # Setup to convert string to activation function
        if activ_fun == 'relu':
            torch_activ_fun = torch.nn.ReLU()
        elif activ_fun == 'tanh':
            torch_activ_fun = torch.nn.Tanh()
        elif activ_fun == 'sigmoid':
            torch_activ_fun = torch.nn.Sigmoid()
        elif activ_fun == 'tanhshrink':
            torch_activ_fun = torch.nn.Tanhshrink()
        elif activ_fun == 'selu':
            torch_activ_fun = torch.nn.SELU()
        else:
            raise ValueError(f'Invalid activ_fun. You passed {activ_fun}')
        # LSTM cell
        if isinstance(lstm_hidden_size, int) and lstm_hidden_size:
            self.lstm = [torch.nn.LSTM(lstm_input_size, lstm_hidden_size, batch_first = True, bidirectional = True).to(device)] # Need to send to device because the cells are in a list
        elif isinstance(lstm_hidden_size, (list, tuple)):
            self.lstm = []
            for idx, size in enumerate(lstm_hidden_size):
                if idx == 0:
                    self.lstm.append(torch.nn.LSTM(lstm_input_size, size, batch_first = True, bidirectional = True).to(device)) # Need to send to device because the cells are in a list
                else:
                    self.lstm.append(torch.nn.LSTM(lstm_hidden_size[idx-1], size, batch_first = True, bidirectional = True).to(device)) # Need to send to device because the cells are in a list
        if 'lstm' in dir(self):
            self.lstm = torch.nn.ModuleList(self.lstm) # Need to transform list into a ModuleList so PyTorch updates and interacts with the weights properly
        # Transforming layers list into OrderedDict with layers + activation
        mylist = list()
        for idx, elem in enumerate(layers):
            mylist.append((f'Linear{idx}', torch.nn.Linear(layers[idx][0], layers[idx][1]) ))
            if idx < len(layers)-1:
                mylist.append((f'{activ_fun}{idx}', torch_activ_fun))
        # OrderedDict into NN
        self.model = torch.nn.Sequential(OrderedDict(mylist))
        self.sigmoid = torch.nn.Sigmoid()
        self.init_weights()

    def init_weights(self) -> None: # Adapted from from https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/linear.py#L105 and https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/rnn.py#L1191
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        for weight in self.parameters():
            if len(weight.shape) > 1:
                torch.nn.init.kaiming_uniform_(weight, a = np.sqrt(5), nonlinearity = 'relu')

    def forward(self, x, classification = False):
        if 'lstm' in dir(self):
            for cell in self.lstm:
                x, (ht, _) = cell(x)
                if cell.bidirectional:
                    x = (x[:, :, :x.shape[2]//2] + x[:, :, x.shape[2]//2:]) / 2 # Average between forward and backward
            if cell.bidirectional:
                to_MLP = (ht[0] + ht[1]) / 2 # Average between forward and backward
            else:
                to_MLP = ht
            out = self.model(to_MLP)
        else:
            out = self.model(x)
        if classification:
            probs = self.sigmoid(out)
            probs = (probs.T / probs.sum(axis=1)).T # Normalizing the probabilties to 1
            return probs
        else:
            return out

def differentiable_MCC(pred, target, weight = None, gamma = 1):
    """
    MCC that is continuous instead of discrete for use as a loss function. See https://doi.org/10.48550/arXiv.2402.17131
    In the binary case, if a prediction is 0.7 and the real label is positive, this prediction will contribute 0.7 to TP and 0.3 to FN
    """
    # These commented-out lines are used in the binary case. The non-commented lines work for all classification cases
    # Note that, for the binary case, pred[:, 0] == 1 - pred[:, 1]
    # p_t = pred[:, 1]*target + (1 - pred[:, 1]) * (1-target) # The probability of a correct prediction, independently of whether the true target is positive or negative
    # TP = (p_t**gamma * target).sum() * weight
    # FN = ((1-p_t)**gamma * target).sum() * weight
    # FP = ((1-p_t)**gamma * (1-target)).sum()
    # TN = (p_t**gamma * (1-target)).sum()
    # MCC = (TP*TN - FP*FN) / torch.sqrt((TP+FP) * (TP+FN) * (TN+FP) * (TN+FN) + 1e-8) # 1e-8 to avoid division by 0, since we can't use if statements
    ######
    if weight is None:
        weight = torch.ones(pred.shape[1]).to(pred.device) # Default: all classes have the same weight
    p_t = torch.zeros(pred.shape[0]).to(pred.device) # The probability of a correct prediction, independently of the true target/true class
    class_counts = torch.zeros(pred.shape[1], dtype = torch.float32).to(pred.device)
    weighted_preds = torch.zeros_like(class_counts).to(pred.device)
    for idx in range(pred.shape[1]):
        p_t += pred[:, idx]**gamma * (target==idx) * weight[idx]
        class_counts[idx] = ( (pred[:, idx]**gamma + (1-pred[:, idx])**gamma) * (target == idx) ).sum() * weight[idx] # A more normal way of calculating this is class_counts[idx] = (target==idx).sum() * weight[idx]. However, this fails to account for gamma, which also changes the class counts
        weighted_preds += (pred[target==idx]**gamma * weight[idx]).sum(axis = 0)
    MCC = ( p_t.sum()*class_counts.sum() - torch.dot(class_counts, weighted_preds) ) / (torch.sqrt( class_counts.sum()**2 - torch.dot(weighted_preds, weighted_preds) ) * torch.sqrt( class_counts.sum()**2 - torch.dot(class_counts, class_counts) ) + 1e-8)
    return -MCC # Negative because we want to maximize the MCC

def _get_best_hyperparameters(final_val_loss):
    """
    A helper function to obtain and format the best hyperparameters after cross-validation of an MLP or RNN model.
    Is called automatically by SPA and shouldn't be called by the user.
    """
    best_idx = np.unravel_index(np.nanargmin(final_val_loss.values), final_val_loss.shape)
    best_LR = final_val_loss.index[best_idx[0]][1]
    best_neurons_str = final_val_loss.columns[best_idx[1]]
    best_act = final_val_loss.index[best_idx[0]][0]
    best_act_loc = final_val_loss.index.get_loc(best_act)
    if len(final_val_loss.index[0]) == 3:
        best_diff_MCC_gamma = final_val_loss.index[best_idx[0]][2]
    else:
        best_diff_MCC_gamma = None
    # Converting the best number of neurons from str to list
    best_neurons = []
    temp_number = []
    temp_tuple = []
    for elem in best_neurons_str:
        if elem in '0123456789':
            temp_number.append(elem)
        elif elem in {',', ')'} and temp_number: # Finished a number. 2nd check because there is a comma right after )
            converted_number = ''.join(temp_number)
            temp_tuple.append( int(converted_number) )
            temp_number = []
        if elem in {')'}: # Also finished a tuple
            best_neurons.append(tuple(temp_tuple))
            temp_tuple = []
    return best_idx, best_LR, best_neurons, best_act, best_act_loc, best_diff_MCC_gamma
