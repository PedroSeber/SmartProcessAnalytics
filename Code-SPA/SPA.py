"""
Original work by Weike (Vicky) Sun vickysun@mit.edu/weike.sun93@gmail.com, https://github.com/vickysun5/SmartProcessAnalytics
Modified by Pedro Seber, https://github.com/PedroSeber/SmartProcessAnalytics
"""
import numpy as np
from dataset_property_new import nonlinearity_assess, collinearity_assess, residual_analysis, nonlinearity_assess_dynamic
import cv_final as cv
from sklearn.preprocessing import StandardScaler
from copy import deepcopy
# To read input files
from os.path import splitext
from pandas import read_excel, read_csv
# To save the results
import json
import pickle
from time import localtime
# Convenience imports
from matplotlib import style
style.use('default')
from os import environ
environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # To hide TF optimization messages
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings('ignore', category = ConvergenceWarning)
warnings.filterwarnings('ignore', category = RuntimeWarning)
import pdb

def main_SPA(main_data, test_data = None, interpretable = False, continuity = False, group_name = None, spectral_data = False,
            plot_interrogation = False, enough_data = False, nested_cv = False, robust_priority = False, dynamic_model = False, lag = [0],
            alpha = 0.01, cat = None, xticks = None, yticks = ['y'], model_name = None, cv_method = None, K_fold = 5, Nr = 10, alpha_num = 20,
            degree = [1, 2, 3], num_outer = 10, K_steps = 1, l1_ratio = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99],
            trans_type = 'auto', select_value = 0.10, RNN_activation = ['relu'], RNN_layers = None, RNN_cell = ['basic'], RNN_batch_size = 1,
            RNN_epoch_overlap = None, RNN_past_steps = 10, RNN_max_checks_without_progress = 50, RNN_learning_rate = 1e-3, RNN_lambda_l2_reg = 1e-3,
            RNN_num_epochs = 200, maxorder = 10, ADAPTx_path = None, ADAPTx_save_path = None, ADAPTx_max_lag = 12, ADAPTx_degrees = [-1, 0, 1]):
    """
    The main SPA function, which calls all other functions needed for model building.

    Parameters
    ----------
    main_data : string
        The path to the file containing your training data.
        The data should be N x (m+1), where the last column contains the predicted variable.
    test_data : string, optional, default = None
        The path to the file containing your test data.
        If None, the main_data is also used as test data (not recommended).
    interpretable : boolean, optional, default = False
        Whether you require the model to be interpretable.
    continuity : boolean, optional, default = False
        Whether you require the model to be continuous, such as for use in optimizers.
    group_name : string or None, optional, default = None
        The path to the file containing group labels for each variable (Nx1).
        Data may be grouped, for example, due to replicated measurements.
        If your data are not grouped, leave as None.
    spectral_data : boolean, optional, default = False
        Whether your data are spectral data.
        Note spectral data force the use of linear models.
    plot_interrogation : boolean, optional, default = False
        Whether SPA should generate plots of the data interrogation results. 
    enough_data : boolean, optional, default = False
        Whether you believe you have enough data to capture the complexities of your system.
    nested_cv : boolean, optional, default = False
        Whether to used nested cross-validation.
        Relevant only when enough_data == False.
    robust_priority : boolean, optional, default = False
        Whether to prioritize robustness over accuracy.
        Relevant only when enough_data == False.
    dynamic_model : boolean, optional, default = False
        Whether to use a dynamic model.
    lag : list of integers, optional, default = [0]
        The lag used when assessing nonlinear dynamics.
        Relevant only when dynamic_model == True.
    alpha : float, optional, default = 0.01
        Significance level when doing statistical tests
    cat : list of int or None, optional, default = None
        Which variables are categorical. None represents no categorical variables.
        e.g.: [1, 0, 0] indicates only the first out of 3 variables is categorical.
    xticks : list of str or None, optional, default = None
        The names used to label x variables in plots generated by SPA.
        If None, SPA uses x1, x2, x3... as default values.
    yticks : list of str, optional, default = ['y']
        A single name to label the y variable in plots generated by SPA.
    model_name : list of str or None, optional, default = None
        The name of the model(s) you want SPA to evaluate.
        Each entry must be in {'OLS', 'ALVEN', 'SVR', 'RF', 'EN', 'SPLS', 'RR', ...
            'PLS', 'DALVEN', 'DALVEN_full_nonlinear', 'RNN', 'SS'}. # TODO: add LASSO?
        If None, SPA determines which models are viable based on the data.
    cv_method : str or None, optional, default = None
        Which cross validation method to use.
        Each entry must be in {'Single', 'KFold', 'MC', 'Re_KFold'} when dynamic_model == False ...
            or {'Single_ordered', 'Timeseries', 'AIC', 'AICc', 'BIC'} when dynamic_model == True.
    K_fold : int, optional, default = 5
        Number of folds used in cross validation.
    Nr : int, optional, default = 10
        Number of CV repetitions used when cv_method in {'MC', 'Re_KFold', 'GroupShuffleSplit'}.
    alpha_num : int, optional, default = 20
        Penalty weight used when model_name in {'RR', 'EN', 'ALVEN', 'DALVEN', 'DALVEN_full_nonlinear'}.
    degree : list of int, optional, default = [1]
        The degrees of nonlinear mapping.
        Relevant only when model_name == 'DALVEN' or 'DALVEN_full_nonlinear'
    num_outer : int, optional, default = 10
        Number of outer loops used in nested CV.
        Relevant only when nested_cv == True.
    K_steps : int, optional, default = 1
        Number of future steps for training and test predictions.
        Relevant only when dynamic_model == True
    l1_ratio : list of floats, optional, default = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99]
        Ratio of L1 penalty to total penalty. When l1_ratio == 1, only the L1 penalty is used.
        Relevant only when model_name in {'EN', 'ALVEN', 'DALVEN', 'DALVEN_full_nonlinear'}
    trans_type : string, optional, default = 'auto'
        The variables to be tested in ALVEN / DALVEN.
        Must be either 'auto' (all variables) or 'poly' (polynomial powers only).
    select_value : float, optional, default = 0.10
        The (p-value or percent) cutoff for ALVEN / DALVEN to select a variable
    RNN_activation : list of str, optional, default = ['relu']
        The activation function(s) used to build an RNN.
        Each entry must be in {'relu', 'tanh', 'sigmoid', 'linear'}.
        If multiple values, all are tested against the validation set ...
            and the best is selected.
        All 'RNN_' parameters are relevant only when model_name == 'RNN'
    RNN_layers : array, optional, default = None
        An array in which each entry is a container with the number of neurons in each layer.
        The length of each container array is the number of hidden layers.
        e.g.: [[512, 256]] tests a single RNN with 2 hidden layers, one with 512 ...
            neurons and the other with 256 neurons. [[512, 256], [256, 256]] tests ...
            2 RNNs, each with 2 hidden layers, and so on.
        If None, is automatically set to [[X_train.shape[1]]] = m (a double list).
    RNN_cell : list of str, optional, default = ['basic']
        The cell type(s) of the RNN.
        Each entry must be in {'basic', 'GNN', 'LSTM'}.
        If multiple values, all are tested against the validation set ...
            and the best is selected.
    RNN_batch_size : int, optional, default = 1
        The batch size used when training the RNN.
    RNN_epoch_overlap : int or None, optional, default = None
        The space between two different training batches.
        If None, there is no overlap, and all batches use different data.
    RNN_past_steps : int, optional, default = 10
        The number of past steps the RNN can use.
    RNN_max_checks_without_progress : int, optional, default = 50
        How many steps without improvement in the validation score ...
            can occur before stopping early.
    RNN_learning_rate : float, optional, default = 1e-3
        The learning rate used when training the RNN.
    RNN_lambda_l2_reg : float, optional, default = 1e-3
        The weight of the L2 regularization penalty.
    RNN_num_epochs : int, optional, default = 200
        The number of RNN training epochs.
    maxorder : int, optional, default = 10
        The maximum state space order used.
        Relevant only when dynamic_model == True and the data are linear. 
    ADAPTx_path : string, optional, default = None
        The path to ADAPTx.
        Relevant only when dynamic_model == True and the data are linear. 
    ADAPTx_save_path : string, optional, default = None
        The path where ADAPTx results will be saved.
        Relevant only when ADAPTx_path is not None.
    ADAPTx_max_lag : int, optional, default = 12
        Maximum number of lags considered during ADAPTx's CVA.
        Relevant only when ADAPTx_path is not None.
    ADAPTx_degrees : list of int, optional, default = [-1, 0, 1]
        Degrees of trend t in ADAPTx
        Relevant only when ADAPTx_path is not None.
    """
    # Loading group (the actual data) from group_name (a path)
    if group_name:
        group = load_file(group_name).flatten()
    else:
        group = None

    # Loading the data
    Data = load_file(main_data)
    X_original = Data[:,:-1]
    y_original = Data[:,-1].reshape(-1,1)
    m = np.shape(X_original)[1]
    N = np.shape(X_original)[0]

    if test_data:
        Test_data = load_file(test_data)
        X_test_original = Test_data[:,:-1]
        y_test_original = Test_data[:,-1].reshape(-1,1)
        N_test = np.shape(X_test_original)[0]
    else:
        X_test_original = None
        y_test_original = None

    if cat is None:
        cat = [0] * (m+1)
    # Ensuring the user didn't pass too many plot labels by mistake
    if isinstance(xticks, (list, tuple)):
        xticks = xticks[:m]
    if isinstance(yticks, (list, tuple)):
        yticks = yticks[:1] # [:1] returns a one-element list, while [0] returns the object


    # Selecting a model
    if model_name is None:
        # Determining nonlinearity and multicollinearity automatically
        nonlinear = nonlinearity_assess(X_original, y_original, plot_interrogation, cat = cat, alpha = alpha, difference = 0.4, xticks = xticks, yticks = yticks, round_number = 0)
        multicollinear = collinearity_assess(X_original, y_original, plot_interrogation, xticks =  xticks, yticks = yticks, round_number = 0)
        if not nonlinear and dynamic_model:
            nonlinear_dynamic = nonlinearity_assess_dynamic(X_original, y_original, plot_interrogation, alpha = alpha, difference = 0.4, xticks = xticks, yticks = yticks, round_number = 0, lag = max(lag))
            if nonlinear_dynamic:
                nonlinear = True

        if nonlinear:
            # Nonlinear, nondynamic models
            if not dynamic_model:
                model_name = ['ALVEN']
                if not enough_data or interpretable:
                    print(f'As {"your data are limited"*(not enough_data)}{" and "*(not(enough_data) and interpretable)}{"you require an interpretable model"*interpretable}, only ALVEN will be used.')
                elif continuity:
                    print('As you have enough data, do not require the model to be interpretable, and require continuity, ALVEN and SVR will be tested')
                    model_name.append('SVR')
                else:
                    print('As you have enough data, do not require the model to be interpretable, and do not require continuity, ALVEN, SVR, and RF will be tested')
                    model_name.append('SVR')
                    model_name.append('RF')
            # Nonlinear, dynamic models
            elif not enough_data or interpretable:
                print(f'As {"your data are limited"*(not enough_data)}{" and "*(not(enough_data) and interpretable)}{"you require an interpretable model"*interpretable}, DALVEN will be used.')
                model_name = ['DALVEN']
            else:
                print('As you have enough data and do not require the model to be interpretable, an RNN will be used.')
                model_name = ['RNN']
        # Linear, nondynamic models
        elif not dynamic_model:
            if not multicollinear:
                print('As there is no significant nonlinearity and multicollinearity in the data, OLS will be used.')
                model_name = ['OLS']
            elif spectral_data:
                print('As you have spectral data, RR and PLS will be used.')
                model_name = ['RR','PLS']
            elif interpretable:
                print('As you require an interpretable model, EN and SPLS will be used.')
                model_name = ['EN','SPLS']
            else:
                print('As your data have significant multicollinearity and you do not require an interpretable model, EN, SPLS, RR, and PLS will be used.')
                model_name = ['EN','SPLS','RR','PLS']
        # Linear dynamic model
        else:
            print('As your data have significant dynamics and multicolinearity, SS will be used.') # Originally CVA, SSARX, and MOSEP
            model_name = ['SS']

    # Cross-Validation Strategy
    if cv_method is None:
        if not dynamic_model:
            if enough_data:
                cv_method = f'Single{"_group"*bool(group_name)}'
                print(f'Single {"grouped CV "*bool(group_name)}{"validation "*(not bool(group_name))}will be used.') # Single grouped CV or single validation set
            elif group_name is None:
                cv_method = 'Re_KFold'
                print(f'{"Nested "*nested_cv}CV with repeated KFold in inner loop {"and one-std rule "*robust_priority}will be used.')
            else:
                cv_method = 'GroupKFold'
                print(f'{"Nested "*nested_cv}GroupKFold {"with one-std rule "*robust_priority}will be used.')
        # Dynamic models
        elif model_name == ['SS']:
            print('MATLAB/ADAPTx packges with information criterion will be used.')
        elif enough_data:    
            cv_method = 'Single_ordered'
            print('Single validation for time series will be used.')
        elif nested_cv:
            cv_method = 'Timeseries'
            print('Cross-validation for time series {"with one-std rule "*robust_priority}will be used.')
        elif robust_priority:
            cv_method = 'BIC'
            print('BIC information criteria will be used.')
        elif X_original.shape[0]//X_original.shape[1]<40:
            cv_method = 'AICc'
            print('AICc information criteria will be used.')
        else:
            cv_method = 'AIC'
            print('AIC information criteria will be used.')

    # Preprocessing the data
    X = deepcopy(X_original)
    y = deepcopy(y_original)
    # Scaling the data
    scaler_x = StandardScaler(with_mean=True, with_std=True)
    scaler_x.fit(X)
    X_scale = scaler_x.transform(X)
    scaler_y = StandardScaler(with_mean=True, with_std=True)
    scaler_y.fit(y)
    y_scale = scaler_y.transform(y)

    if X_test_original is not None:
        X_test = deepcopy(X_test_original)
        y_test = deepcopy(y_test_original)
        X_test_scale = scaler_x.transform(X_test)
        y_test_scale = scaler_y.transform(y_test)
    else:
        X_test = X
        y_test = y
        X_test_scale = X_scale
        y_test_scale = y_scale

    # Model fitting
    fitting_result = {}

    if 'OLS' in model_name:
        from regression_models import OLS_fitting
        final_model, model_params, mse_train, mse_test, yhat_train, yhat_test = OLS_fitting(X_scale, y_scale, X_test_scale, y_test_scale)
        fitting_result['OLS'] = {'final_model':final_model, 'model_params':model_params, 'mse_train':mse_train, 'mse_test':mse_test, 'yhat_train':yhat_train, 'yhat_test':yhat_test}
        selected_model = 'OLS'

    # Non-dynamic models
    if any(temp_model in model_name for temp_model in {'ALVEN', 'SVR', 'RF', 'RR', 'PLS', 'EN', 'PLS', 'SPLS'}) and 'OLS' not in model_name: # TODO: how do we compare OLS with the other models if OLS doesn't have validation scores?
        # Static / traditional CV
        if not nested_cv:
            MSE_val = np.empty(len(model_name)) * np.nan
            temp_fitting_result = {}
            for index, this_model in enumerate(model_name):
                if this_model in {'ALVEN', 'SVR', 'RF', 'RR', 'PLS', 'EN', 'PLS', 'SPLS'}: # There may be dynamic models if the user passed model_name manually
                    print(f'Running model {this_model}', end = '\r')
                    fitting_result[this_model], MSE_val[index] = run_cv_nondynamic(this_model, X, y, X_scale, y_scale, X_test, y_test, X_test_scale, y_test_scale,
                            cv_method, group, K_fold, Nr, alpha_num, l1_ratio, robust_priority, degree, trans_type, cat[-1], select_value)
                    print(f'Completed model {this_model}')
            local_selected_model = model_name[np.nanargmin(MSE_val)]

        # Nested CV
        else: 
            if group_name is None:
                from sklearn.model_selection import train_test_split
                MSE_val = np.empty((len(model_name), num_outer)) * np.nan

                for index_out in range(num_outer):
                    print(f'Beginning nested CV loop {index_out+1} out of {num_outer}', end = '/r')
                    X_nest, X_nest_val, y_nest, y_nest_val = train_test_split(X, y, test_size=1/K_fold, random_state = index_out)
                    X_nest_scale, X_nest_scale_val, y_nest_scale, y_nest_scale_val = train_test_split(X_scale, y_scale, test_size=1/K_fold, random_state= index_out)
                    for index, this_model in enumerate(model_name):
                        if this_model in {'ALVEN', 'SVR', 'RF', 'RR', 'PLS', 'EN', 'PLS', 'SPLS'}: # There may be dynamic models if the user passed model_name manually
                            MSE_val[index, index_out] = run_cv_nondynamic(this_model, X_nest, y_nest, X_nest_scale, y_nest_scale, X_nest_val, y_nest_val, X_nest_scale_val,
                                    y_nest_scale_val, cv_method, group, K_fold, Nr, alpha_num, l1_ratio, robust_priority, degree, trans_type, cat[-1], select_value, True)
            else:
                from sklearn.model_selection import LeaveOneGroupOut
                MSE_val = np.empty((len(model_name), len(np.unique(group)))) * np.nan
                RMSE_val = np.empty((len(model_name), len(np.unique(group)))) * np.nan
                logo = LeaveOneGroupOut()

                for index_out, (train, val) in enumerate( logo.split(X, y.flatten(), groups = group.flatten()) ):
                    print(f'Beginning nested CV loop {index_out+1}', end = '/r')
                    for index, this_model in enumerate(model_name):
                        if this_model in {'ALVEN', 'SVR', 'RF', 'RR', 'PLS', 'EN', 'PLS', 'SPLS'}: # There may be dynamic models if the user passed model_name manually
                            MSE_val[index, index_out] = run_cv_nondynamic(this_model, X[train], y[train], X_scale[train], y_scale[train], X[val], y[val], X_scale[val], y_scale[val],
                                    cv_method, group[train], K_fold, Nr, alpha_num, l1_ratio, robust_priority, degree, trans_type, cat[-1], select_value, True)

            # Nested CV MSE results
            time_now = '-'.join([str(elem) for elem in localtime()[:6]]) # YYYY-MM-DD-hh-mm-ss
            import matplotlib.pyplot as plt
            plt.figure()
            pos = [i+1 for i in range(len(model_name))]
            ax = plt.subplot(111)
            plt.violinplot(np.transpose(MSE_val))
            ax.set_xticks(pos)
            ax.set_xticklabels(model_name)
            ax.set_title('Testing MSE distribution using nested CV')
            plt.savefig(f'MSE_violin_plot_{time_now}.png')
            RMSE_val = np.sqrt(MSE_val)
            plt.figure()
            pos = [i+1 for i in range(len(model_name))]
            ax = plt.subplot(111)
            plt.violinplot(np.transpose(RMSE_val))
            ax.set_xticks(pos)
            ax.set_xticklabels(model_name)
            ax.set_title('Testing RMSE distribution using nested CV')
            plt.savefig(f'RMSE_violin_plot_{time_now}.png')

            # Final model fitting
            local_selected_model = model_name[np.nanargmin(np.mean(MSE_val, axis = 1))]
            fitting_result[local_selected_model], _ = run_cv_nondynamic(local_selected_model, X, y, X_scale, y_scale, X_test, y_test, X_test_scale, y_test_scale,
                    cv_method, group, K_fold, Nr, alpha_num, l1_ratio, robust_priority, degree, trans_type, cat[-1], select_value)

    # Dynamic models
    if 'RNN' in model_name:
        import timeseries_regression_RNN as t_RNN
        if RNN_layers is None:
            RNN_layers = [[m]]

        print('Running model RNN', end = '\r')        
        if robust_priority and cv_method in {'AIC', 'AICc'}:
            print(f'Note: BIC is recommended for robustness, but you selected {cv_method}.')
        RNN_hyper, RNN_model, yhat_train_RNN, yhat_val_RNN, yhat_test_RNN, mse_train_RNN, mse_val_RNN, mse_test_RNN = cv.CV_mse('RNN', X_scale,
                y_scale, X_test_scale, y_test_scale, cv_method, K_fold, Nr, cell_type = RNN_cell, group = group, activation = RNN_activation, RNN_layers = RNN_layers,
                num_steps = RNN_past_steps, batch_size = RNN_batch_size, epoch_overlap = RNN_epoch_overlap, learning_rate = RNN_learning_rate,
                lambda_l2_reg = RNN_lambda_l2_reg, num_epochs = RNN_num_epochs, max_checks_without_progress = RNN_max_checks_without_progress, robust_priority = robust_priority)

        fitting_result['RNN'] = {'model_hyper': RNN_hyper, 'final_model': RNN_model, 'mse_train': mse_train_RNN, 'mse_val': mse_val_RNN, 'mse_test': mse_test_RNN,
                'yhat_train': yhat_train_RNN, 'yhat_val': yhat_val_RNN, 'yhat_test': yhat_test_RNN, 'MSE_val': MSE_val}
        print('Finished model RNN')

    if 'DALVEN' in model_name:
        print('Running model DALVEN', end = '\r')        
        fitting_result['DALVEN'] = run_DALVEN('DALVEN', X, y, X_test, y_test, cv_method, alpha_num, lag, degree, K_fold, Nr, robust_priority, l1_ratio, trans_type, cat[-1])
        print('Finished model DALVEN')

    if 'DALVEN_full_nonlinear' in model_name:
        print('Running model DALVEN_full_nonlinear', end = '\r')        
        fitting_result['DALVEN_full_nonlinear'] = run_DALVEN('DALVEN_full_nonlinear', X, y, X_test, y_test, cv_method, alpha_num, lag, degree, K_fold, Nr,
                    robust_priority, l1_ratio, trans_type, cat[-1])
        print('Finished model DALVEN_full_nonlinear')

    if 'SS' in model_name: # SS
        import timeseries_regression_matlab as t_matlab
        # MATLAB
        matlab_params, matlab_myresults, matlab_MSE_train, matlab_MSE_val, matlab_MSE_test, matlab_y_predict_train, matlab_y_predict_val, matlab_y_predict_test,\
                matlab_train_error, matlab_val_error, matlab_test_error = t_matlab.timeseries_matlab_single(X, y, X_test, y_test, train_ratio = 1,
                maxorder = maxorder, mynow = 1, steps = K_steps, plot = plot_interrogation)
        local_selected_model = matlab_params['method'][0]
        fitting_result[local_selected_model] = {'model_hyper': matlab_params, 'final_model': matlab_myresults, 'mse_train': matlab_MSE_train, 'mse_val':matlab_MSE_val,
                'mse_test': matlab_MSE_test, 'yhat_train': matlab_y_predict_train, 'yhat_val': matlab_y_predict_val, 'yhat_test': matlab_y_predict_test}
        # ADAPTx
        if ADAPTx_path:
            import timeseries_regression_Adaptx as t_Adaptx
            Adaptx_optimal_params, Adaptx_myresults, Adaptx_MSE_train, Adaptx_MSE_val, Adaptx_MSE_test, Adaptx_y_predict_train, Adaptx_y_predict_val, Adaptx_y_predict_test,\
                    Adaptx_train_error, Adaptx_val_error, Adaptx_test_error = t_Adaptx.Adaptx_matlab_single(X, y, ADAPTx_save_path, ADAPTx_path, X_test, y_test, train_ratio = 1,
                    max_lag = ADAPTx_max_lag, mydegs = ADAPTx_degrees, mynow = 1, steps = K_steps, plot = plot_interrogation) 
            fitting_result['ADAPTx'] = {'model_hyper': Adaptx_optimal_params, 'final_model': Adaptx_myresults, 'mse_train': Adaptx_MSE_train, 'mse_val': Adaptx_MSE_val,
                    'mse_test': Adaptx_MSE_test, 'yhat_train': Adaptx_y_predict_train, 'yhat_val': Adaptx_y_predict_val, 'yhat_test': Adaptx_y_predict_test}

    # Finding the best model
    for idx, entry in enumerate(fitting_result):
        if idx == 0 or fitting_result[entry]['mse_val'] < fitting_result[selected_model]['mse_val']:
            selected_model = entry
    # Catching wrong model names
    for model in model_name:
        if model not in fitting_result:
            warnings.warn(f'{model} is not a valid model name, so it was ignored.')
    if 'selected_model' not in locals():
        raise UnboundLocalError(f'You input {model_name} for model_name, but that is not a valid name.')

    # Residual analysis + test for dynamics in the residual
    fitting_result[selected_model]['yhat_train_nontrans'] = scaler_y.inverse_transform(np.atleast_2d(fitting_result[selected_model]['yhat_train']))
    fitting_result[selected_model]['yhat_train_nontrans_mean'] = np.mean(fitting_result[selected_model]['yhat_train_nontrans'])
    fitting_result[selected_model]['yhat_train_nontrans_stdev'] = np.std(fitting_result[selected_model]['yhat_train_nontrans'])
    fitting_result[selected_model]['MSE_train_nontrans'] = np.sum( (fitting_result[selected_model]['yhat_train_nontrans'] - y)**2 )/y.shape[0]
    fitting_result[selected_model]['RMSE_train_nontrans'] = np.sqrt(fitting_result[selected_model]['MSE_train_nontrans'])
    fitting_result[selected_model]['yhat_test_nontrans'] = scaler_y.inverse_transform(np.atleast_2d(fitting_result[selected_model]['yhat_test']))
    fitting_result[selected_model]['yhat_test_nontrans_mean'] = np.mean(fitting_result[selected_model]['yhat_test_nontrans'])
    fitting_result[selected_model]['yhat_test_nontrans_stdev'] = np.std(fitting_result[selected_model]['yhat_test_nontrans'])
    fitting_result[selected_model]['MSE_test_nontrans'] = np.sum( (fitting_result[selected_model]['yhat_test_nontrans'] - y_test)**2 )/y_test.shape[0]
    fitting_result[selected_model]['RMSE_test_nontrans'] = np.sqrt(fitting_result[selected_model]['MSE_test_nontrans'])
    if len(y_test.squeeze()) >= 4: # TODO: residuals with small lengths lead to errors when plotting ACF. Need to figure out why
        if 'DALVEN' in selected_model: # The first "lag" entries are removed from yhat, so we need to remove them from X and y
            lag = fitting_result[selected_model]['model_hyper']['lag']
        else:
            lag = 0
        _, dynamic_test_result = residual_analysis(X_test[lag:], y_test[lag:], fitting_result[selected_model]['yhat_test_nontrans'], plot = plot_interrogation, alpha = alpha, round_number = selected_model)
        if dynamic_test_result and not(dynamic_model) and selected_model not in {'RNN', 'DALVEN', 'DALVEN_full_nonlinear', 'ADAPTx'}: # TODO: Get the names of the MATLAB models and add them here
            print('A residual analysis found dynamics in the system. Please run SPA again with dynamic_model = True')
            print('Note that specifying a non-dynamic model will override the dynamic_model flag')

    # Setup for saving
    # jsons do not work with numpy arrays - converting to list
    fr2 = fitting_result.copy()
    for model in fr2.keys():
        del fr2[model]['final_model'] # Models aren't convertible to json
        for top_key, top_value in fr2[model].items():
            if isinstance(fr2[model][top_key], dict):
                for key, value in fr2[model][top_key].items():
                    if isinstance(value, np.ndarray):
                        fr2[model][top_key][key] = value.tolist()
            elif isinstance(top_value, np.ndarray):
                fr2[model][top_key] = top_value.squeeze().tolist()
    # Saving as a pickled file
    with open(f'SPA_results_{time_now}.p', 'wb') as f:
        pickle.dump(fitting_result, f)
    # Saving as a json file
    with open(f'SPA_results_{time_now}.json', 'w') as f:
        json.dump(fr2, f, indent = 4)

    print(f'The best model is {selected_model}. View its results via fitting_result["{selected_model}"] or by opening the SPA_results json/pickle files.')
    return fitting_result, selected_model

def load_file(filename):
    """
    Used by SPA to load data files.
    """
    _, ext = splitext(filename)
    if ext == '.txt':
        for separator in (' ', ',', '\t', ';'): # Testing random separators
            my_file = read_csv(filename, header = None, sep = separator).values
            if my_file.shape[-1] > 1: # We likely found the separator
                break
    elif ext == '.csv':
        my_file = read_csv(filename, header = None, sep = ',').values
    elif ext == '.tsv':
        my_file = read_csv(filename, header = None, sep = '\t').values
    elif ext in {'.xls', '.xlsx'}:
        my_file = read_excel(filename, header = None).values
    else:
        raise ValueError(f'Please provide a filename with extension in {{.txt, .csv, .tsv, .xls, .xlsx}}. You passed {filename}')
    return my_file

def run_cv_nondynamic(model_index, X_train, y_train, X_train_scaled, y_train_scaled, X_test, y_test, X_test_scaled, y_test_scaled, cv_method,
                        group, K_fold, Nr, alpha_num, l1_ratio, robust_priority, degree, trans_type, use_cross_entropy, select_value, for_validation = False):
    """
    Runs a nondynamic model for CV or final-run purposes. Automatically called by SPA.

    Parameters
    ----------
    model_index to select_value
        Automatically called by SPA based on what was passed to main_SPA()
    for_validation : bool, optional, default = False
        Whether the run is done to validate a model (to determine ...
            the best model) or test the best model.
        This changes a little the syntax and values returned, but not the logic.
    """
    if for_validation:
        # For the sake of clarity
        X_val, y_val = X_test, y_test
        X_val_scaled, y_val_scaled = X_test_scaled, y_test_scaled

        if model_index == 'ALVEN':
            _, _, _, _, mse_val, _, _, _, _ = cv.CV_mse(model_index, X_train, y_train, X_val, y_val, cv_type = cv_method,
                    group = group, K_fold = K_fold, Nr = Nr, alpha_num = alpha_num, l1_ratio = l1_ratio, label_name = True,
                    robust_priority = robust_priority, degree = degree, trans_type = trans_type, use_cross_entropy = use_cross_entropy, select_value = select_value)
        elif model_index == 'SVR' or model_index == 'RF':
            _, _, _, mse_val, _, _, _ = cv.CV_mse(model_index, X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, X_train, y_train, cv_type = cv_method,
                    group = group, K_fold = K_fold, Nr = Nr, alpha_num = alpha_num, robust_priority = robust_priority)
        else:
            _, _, _, _, mse_val, _, _, _ = cv.CV_mse(model_index, X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, X_train, y_train, cv_type = cv_method,
                    group = group, K_fold = K_fold, Nr = Nr, alpha_num = alpha_num, l1_ratio = l1_ratio, robust_priority = robust_priority)
        return mse_val
    else:
        if model_index == 'ALVEN':
            model_hyper, final_model, model_params, mse_train, mse_test, yhat_train, yhat_test, mse_val, final_list = cv.CV_mse(model_index, X_train, y_train, X_test, y_test,
                    cv_type = cv_method, group = group, K_fold = K_fold, Nr = Nr, alpha_num = alpha_num, l1_ratio = l1_ratio, label_name = True,
                    robust_priority = robust_priority, degree = degree, trans_type = trans_type, use_cross_entropy = use_cross_entropy, select_value = select_value)
            fitting_result = {'model_hyper':model_hyper, 'final_model':final_model, 'model_params':model_params, 'mse_train':mse_train, 'mse_val':mse_val,
                    'mse_test':mse_test, 'yhat_train':yhat_train, 'yhat_test':yhat_test, 'final_list':final_list}
        elif model_index == 'SVR' or model_index == 'RF':
            model_hyper, final_model, mse_train, mse_test, yhat_train, yhat_test, mse_val = cv.CV_mse(model_index, X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, X_train, y_train,
                    cv_type = cv_method, group = group, K_fold = K_fold, Nr = Nr, alpha_num = alpha_num, robust_priority = robust_priority)
            fitting_result = {'model_hyper':model_hyper, 'final_model':final_model, 'mse_train':mse_train, 'mse_val':mse_val, 'mse_test':mse_test,
                    'yhat_train':yhat_train, 'yhat_test':yhat_test}
        else:
            model_hyper, final_model, model_params, mse_train, mse_test, yhat_train, yhat_test, mse_val = cv.CV_mse(model_index, X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, X_train, y_train,
                    cv_type = cv_method, group = group, K_fold = K_fold, Nr = Nr, alpha_num = alpha_num, l1_ratio = l1_ratio, robust_priority = robust_priority)
            fitting_result = {'model_hyper':model_hyper, 'final_model':final_model, 'model_params':model_params, 'mse_train':mse_train, 'mse_val': mse_val,
                    'mse_test':mse_test, 'yhat_train':yhat_train, 'yhat_test':yhat_test}
        return fitting_result, mse_val

def run_DALVEN(model_name, X, y, X_test, y_test, cv_method, alpha_num, lag, degree, K_fold, Nr, robust_priority, l1_ratio, trans_type, use_cross_entropy):
    """
    Runs DALVEN or DALVEN_full_nonlinear. Automatically called by SPA.

    Parameters
    ----------
    model_name to use_cross_entropy
        Automatically called by SPA based on what was passed to main_SPA()
    """
    if 'IC' in cv_method:
        if robust_priority and cv_method != 'BIC':
            print(f'Note: BIC is recommended for robustness, but you selected {cv_method}.')
        mystring = 'IC_optimal'
    else:
        mystring = 'mse_val'
    DALVEN_hyper, DALVEN_model, DALVEN_params, mse_train_DALVEN, mse_test_DALVEN, yhat_train_DALVEN, yhat_test_DALVEN, MSE_v_DALVEN, final_list = cv.CV_mse(model_name,
            X, y, X_test, y_test, cv_method, K_fold, Nr, alpha_num = alpha_num, lag = lag, degree = degree, l1_ratio = l1_ratio, label_name = True,
            trans_type = trans_type, robust_priority = robust_priority, use_cross_entropy = use_cross_entropy)

    return {'model_hyper': DALVEN_hyper,'final_model': DALVEN_model, 'model_params': DALVEN_params , 'mse_train': mse_train_DALVEN, mystring: MSE_v_DALVEN,
            'mse_test': mse_test_DALVEN, 'yhat_train': yhat_train_DALVEN, 'yhat_test': yhat_test_DALVEN, 'final_list': final_list}
