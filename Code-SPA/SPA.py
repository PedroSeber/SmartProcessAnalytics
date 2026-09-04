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
from torch import save as torchsave
from collections import OrderedDict
# Convenience imports
from matplotlib import style
style.use('default')
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings('ignore', category = ConvergenceWarning)
warnings.filterwarnings('ignore', category = RuntimeWarning)

def main_SPA(main_data, main_data_y = None, test_data = None, test_data_y = None, scale_X = True, scale_y = True, interpretable = False, continuity = False,
            group_name = None, plot_interrogation = False, nested_cv = False, robust_priority = False, dynamic_model = False, lag = [0], min_lag = 0,
            significance = 0.05, cat = None, classification = False, xticks = None, yticks = ['y'], model_name = None, cv_method = None, K_fold = 5, Nr = 10,
            num_outer = 10, l1_ratio = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99], alpha = 20, SPLS_K = None, SPLS_eta = None, degree = [1, 2, 3],
            trans_type = 'all', LCEN_cutoff = 4e-2, LCEN_interaction = True, LCEN_transform_y = False, RF_n_estimators = [10, 25, 50, 100, 200, 300],
            RF_max_depth = [2, 3, 5, 10, 15, 20, 40], RF_min_samples_leaf = [0.001, 0.01, 0.02, 0.05, 0.1], RF_n_features = [0.1, 0.25, 0.333, 0.5, 0.667, 0.75, 1.0],
            SVM_gamma = None, SVM_C = [0.001, 0.01, 0.1, 1, 10, 50, 100, 500], SVM_epsilon = [0.01, 0.02, 0.03, 0.05, 0.08, 0.09, 0.1, 0.15, 0.2, 0.3], activation = ['relu'],
            MLP_layers = None, RNN_layers = None, batch_size = 32, learning_rate = [1e-2, 5e-3], weight_decay = 0, l1_penalty_factor = 0, n_epochs = 100,
            class_weight = None, scheduler = 'plateau', scheduler_mode = 'min', scheduler_factor = 0.5, scheduler_patience = 10, scheduler_min_LR = 1/16,
            scheduler_last_epoch = None, scheduler_warmup = 10, val_loss_file = None, expand_hyperparameter_search = False, verbosity_level = 2):
    """
    The main SPA function, which calls all other functions needed for model building.

    Parameters
    ----------
    main_data : string
        The path to the file containing your training data.
        The data should be N x (m+1), where the last column contains the predicted variable.
        Alternatively, the data can be N x m, and contain only inputs. The predicted variable is ...
            passed to SPA through the main_data_y variable.
    main_data_y : string, optional, default = None
        The path to the file containing the outputs of the training data.
        These data should have the same number of rows as main_data.
        If None, the last column of main_data is treated as the output.
    test_data : string, optional, default = None
        The path to the file containing your test data.
        If None, the main_data is also used as test data (not recommended).
    test_data_y : string, optional, default = None
        The path to the file containing the outputs of the test data.
        These data should have the same number of rows as test_data.
        If None, the last column of test_data is treated as the output.
    scale_X : boolean, optional, default = True
        Whether to scale the X data.
    scale_y : boolean, optional, default = True
        Whether to scale the y data.
        Relevant only when doing regression and not classification.
    interpretable : boolean, optional, default = False
        Whether you require the model to be interpretable.
    continuity : boolean, optional, default = False
        Whether you require the model to be continuous, such as for use in optimizers.
    group_name : string or None, optional, default = None
        The path to the file containing group labels for each variable (Nx1).
        Data may be grouped, for example, due to replicated measurements.
        If your data are not grouped, leave as None.
    plot_interrogation : boolean, optional, default = False
        Whether SPA should generate plots of the data interrogation results. 
    nested_cv : boolean, optional, default = False
        Whether to used nested cross-validation.
        Nested CV runs CV multiple times, each time using a different part of the set for final testing.
        This increases the selected model's robustness but requires much more time for validation.
    robust_priority : boolean, optional, default = False
        If True, the selected model will be the model with the lowest number of features ...
            within one stdev of the model with the lowest validation MSE.
        If False (default), the model with the lowest validation MSE is selected.
    dynamic_model : boolean, optional, default = False
        Whether to use a dynamic model.
    lag : list of integers, optional, default = [0]
        The lag used when running an autoregressive LCEN.
        Relevant only when dynamic_model == True or model_name == 'LCEN'.
        For correct results, lag should be increased by min_lag (see below).
    min_lag : integer, optional, default = 0
        The minimum lag used in autoregressive LCEN. Should be smaller than the smallest lag.
        Useful when making N-points ahead predictions, in which case min_lag should be set to N-1 and ...
            each entry in lag should be increased by N-1.
            (e.g.: the default min_lag = 0 is used to make predictions 1 point ahead; a min_lag = 5 is
            used to make predictions 6 points ahead, ignoring the first 5 points ahead when training.)
    significance : float, optional, default = 0.05
        Significance level when doing statistical tests
    cat : list of int or None, optional, default = None
        Which variables are categorical. None represents no categorical variables.
        e.g.: [1, 0, 0] indicates only the first out of 3 variables is categorical.
        Relevant only when model_name is None; used to determine the data's nonlinearity.
    classification : bool, optional, default = False
        Whether to train a model for a classification (True) or regression (False) task.
        Note that SPA will automatically change the model architectures appropriately ...
            without changes in model_name. For example, model_name == ['OLS'] with classification == True ...
            trains a multinomial logistic regression model, but it is still called 'OLS'.
    xticks : list of str or None, optional, default = None
        The names used to label x variables in plots generated by SPA.
        If None, SPA uses x1, x2, x3... as default values.
    yticks : list of str, optional, default = ['y']
        A single name to label the y variable in plots generated by SPA.
    model_name : list of str or None, optional, default = None
        The name of the model(s) you want SPA to evaluate.
        Each entry must be in {'OLS', 'LCEN', 'SVM', 'RF', 'GBDT', 'AdaB', 'EN', 'SPLS', 'PLS', ...
            'MLP', 'RNN'}.
        If None, SPA determines which model architectures are viable based on the data.
    cv_method : str or None, optional, default = None
        Which cross validation method to use.
        Each entry must be in {'Single', 'KFold', 'MC', 'Re_KFold'} when dynamic_model == False ...
            or {'Single_ordered', 'Timeseries', 'AIC', 'AICc', 'BIC'} when dynamic_model == True.
    K_fold : int, optional, default = 5
        Number of folds used in cross validation.
    Nr : int, optional, default = 10
        Number of CV repetitions used when cv_method in {'MC', 'Re_KFold', 'GroupShuffleSplit'}.
    num_outer : int, optional, default = 10
        Number of outer loops used in nested CV.
        Relevant only when nested_cv == True.
    l1_ratio : list of floats, optional, default = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99]
        Ratio of L1 penalty to total penalty. When l1_ratio == 1, only the L1 penalty is used.
        Relevant only when at least one of {'EN', 'LCEN'} is in model_name
    alpha : int or list of floats, optional, default = 20
        The weight of the L1 or L2 regularizations used when at least one of {'EN', 'LCEN'} is in model_name
        The regularization term is alpha*l1_ratio*||w||_1 + 0.5*alpha*(1 - l1_ratio)*||w||^2_2 or alpha*||w||^2_2
        If int, a list of alpha+1 values equal to np.concatenate(([0], np.logspace(-4.3, 0, kwargs['alpha']))) is generated
    SPLS_K : list of integers, optional, default = None
        A list of integers representing the maximum number of latent variables for Sparse PLS.
        If None, see the SPLS section in cv_final.py for the defaults (depends on the size of X)
    SPLS_eta : list of floats between 0 and 1, optional, default = None
        A list of values used for the sparsity tuning parameter of Sparse PLS.
        Every value should be between 0 and 1, where 0 is equivalent to PLS.
        If None, defaults to np.linspace(0, 1, 20, endpoint = False)
    degree : list of int, optional, default = [1, 2, 3]
        The maximum degree of the X data transforms for the LCEN algorithm.
        Relevant only when 'LCEN' is in model_name
    trans_type : str in {'all', 'poly', 'simple_interaction'}, optional, default == 'all'
        Whether to include all transforms (polynomial, log, sqrt, and inverse), only polynomial transforms (and, ...
            optionally, interactions), or just interactions.
        The log, sqrt, and inverse terms never include interactions among the same transform type (such as ln(x1)*ln(x4)), but ...
            include some interactions among each other for the same variable (such as ln(x0)*1/x0, x0*sqrt(x0), etc.).
    LCEN_cutoff : float, optional, default = 4e-2
        The minimum absolute value a scaled coefficient needs to have to not be removed by the clip steps of the LCEN algorithm
        Relevant only when 'LCEN' is in model_name
    LCEN_interaction : bool, optional, default = True
        Whether to also include interactions between different variables (such as x1*x3 or x0*x5^2) in the transformed features.
        Note that, if trans_type == 'simple_interaction', this variable must be set to True
        Relevant only when 'LCEN' is in model_name
    LCEN_transform_y : bool, optional, default = False
        Whether to also perform nonlinear transforms on the y variable.
        Relevant only when 'LCEN' in model_name and lag > 0.
    RF_n_estimators : list of integers, optional, default = [10, 25, 50, 100, 200, 300]
        The number of trees in the random forest.
        Relevant only when 'RF' in model_name.
    RF_max_depth : list of integers, optional, default = [2, 3, 5, 10, 15, 20, 40]
        The maximum depth of each tree.
        Relevant only when 'RF' in model_name.
    RF_min_samples_leaf : list of floats between 0 and 1, optional, default = [0.001, 0.01, 0.02, 0.05, 0.1]
        The minimum fraction of samples that must be available for each leaf when splitting at a node.
        Relevant only when 'RF' in model_name
    RF_n_features : list of floats between 0 and 1, optional, default = [0.1, 0.25, 0.333, 0.5, 0.667, 0.75, 1.0]
        The fraction of features available for each tree. Increasing this value makes each individual tree more powerful, ...
            but also makes the trees more correlated with each other.
        Relevant only when 'RF' in model_name.
    SVM_gamma : list of floats >= 0 or str in {'scale', 'auto'}, optional, default = None
        Kernel coefficient for the 'rbf', 'poly', and 'sigmoid' kernels
        If 'scale', gamma = 1 / (X.shape[1] * X.var())
        If 'auto', gamma = 1 / X.shape[1]
        If None, gamma = (1 / X.shape[1]) * [1/50, 1/10, 1/5, 1/2, 1, 2, 5, 10, 50]
        Relevant only when 'SVM' in model_name.
    SVM_C : list of floats > 0, optional, default = [0.001, 0.01, 0.1, 1, 10, 50, 100, 500]
        Parameter inversely proportional to the strength of the regularization.
        Relevant only when 'SVM' in model_name.
    SVM_epsilon : list of floats >= 0, optional, default = [0.01, 0.02, 0.03, 0.05, 0.08, 0.09, 0.1, 0.15, 0.2, 0.3]
        Epsilon-tube within which no penalty is associated in the training loss function.
        Relevant only when 'SVM' in model_name.
    activation : list of str, optional, default = ['relu']
        The activation function(s) used to build ANNs. Each entry must be in {'relu', 'tanh', 'sigmoid', 'tanhshrink', 'selu'}.
        If multiple values, all are cross-validated and the best is selected.
    MLP_layers: array of arrays of ints, optional, default = None
        An array of arrays in which each entry is a container with the number of neurons in each layer, using the ...
            torch.nn.Linear() formatting. The length of each inner container is the number of hidden layers. # TODO: change the input to something more user friendly, such as a direct list?
    RNN_layers : int or array of ints, optional, default = None
        An array in which each entry is the number of neurons in each LSTM hidden layer, such that the length ...
            of the array is the number of LSTM cells.
        e.g.: 512 or [512] tests a single LSTM cell with 512 hidden-layer neurons. [512, 256] tests a single RNN with ...
            2 LSTM cells, one with 512 hidden-layer neurons and the other with 256 neurons, and so on.
        If None, is automatically set to X_train.shape[1], leading to a single LSTM cell.
    batch_size : int, optional, default = 32
        The batch size used when training ANNs.
    learning_rate : array of floats, optional, default = [1e-2, 5e-3]
        The learning rate (LR) used when training GBDTs or ANNs.
    weight_decay : float >= 0, optional, default = 0
        Weight penalty used with the AdamW optimizer when training ANNs.
        Equivalent to L2 regularization.
    l1_penalty_factor : float >= 0, optional, default = 0
        The penalty factor that multiplies the group L1 norm when training ANNs.
    n_epochs : int, optional, default = 100
        The number of ANN training epochs.
    class_weight : array, optional, default = None
        The weights of each class in the loss function. Relevant only when classification == True.
        If None, all class weights are set to 1.
    scheduler : str, optional, default = 'plateau'
        The learning rate scheduler used when training ANNs.
        Must be in {'plateau', 'cosine', 'lambda', 'step', 'multistep', 'exponential'}
    scheduler_mode : str, optional, default = 'min'
        One of 'min' or 'max'. When 'min', assumes that the loss function should be minimized, and vice-versa.
        Used when scheduler == 'plateau'
    scheduler_factor : float or lambda function, optional, default = 0.5
        For the plateau, (multi)step, and exponential schedulers, a float. The value the LR is multiplied with for ...
            reduction when the validation loss plateaus (plateau) or after scheduler_patience epochs (step).
            In other words, new_LR = old_LR * scheduler_factor.
        For the lambda scheduler, a lambda function of the form "lambda epoch: LR adjustment".
    scheduler_patience : integer or list of integers, optional, default = 10
        For the plateau and step schedulers, an integer representing how many epochs must pass before another reduction in LR.
        For the multistep scheduler, a list of integers. LR will be reduced when epoch == each entry of that list.
    scheduler_min_lr : float < 1, optional, default = 1/16
        The lowest LR value the scheduler may set based on a fraction of the starting LR value.
        For example, if scheduler_min_lr is 0.2 and the starting LR is 0.05, the minimum LR would be 0.01.
    scheduler_last_epoch : integer, optional, default = None
        For the cosine scheduler, the last epoch with a LR reduction.
        If None, is set to n_epochs - 30.
        If a negative value, is set to n_epochs - value.
    scheduler_warmup : integer, optional, default = 10:
        For the cosine scheduler, the warmup period in epochs.
        During warmup, the LR starts at 0 and increases linearly to the specified LR.
    val_loss_file : str, optional, default = None
        The path to a .csv file containing CV losses generated by SPA when training ANNs.
            Used to restart a CV procedure. If None, SPA starts from zero.
    expand_hyperparameter_search : str or False, optional, default = False
        Whether to cross-validate more hyperparameters after the initial CV is done. This occurs if the best hyperparameters ...
             are in the extreme tested during CV (for example, the best LR is also the highest value tested).
        If 'grid', all possible new combinations of user-input hyperparameters are tested, which is slower but more complete.
        If 'single', only the immediate step beyond the best hyperparameters is tested, which is faster but less complete.
        For example, if the best hyperparameters are an intermediate layer configuration and the smallest LR, 'grid' would test ...
            a smaller LR with ALL layer configurations, while 'single' would do the same with only the best layer configuration.
            If the best hyperparameters are the largest layer configuration and an intermediate LR, 'grid' would test a ...
            larger layer configuration with ALL LR values, while 'single' would do the same with only the best LR value.
    verbosity_level : int >= 0, optional, default = 2
        An interger representing how verbose SPA should be.
        0: Nothing at all is printed (not recommended).
        1: Only the test set results of the final model, what choices SPA made automatically, and what model is currently being trained are printed.
        2: All in level 1 and progress on the current training (such as CV folds for most models) are printed. [Default]
        3: All in level 2 and additional progress on the current training for MLPs and RNNs, final feature selection information for LCEN, and nested validation progress. Equal to 2 if these model architectures or nested validation is not used.
        4: All in level 3 and progress on each epoch of training for MLPs and RNNs. Equal to 2 if these model architectures are not used.
    """
    # Loading group (the actual data) from group_name (a path)
    if group_name:
        group = load_file(group_name).flatten()
    else:
        group = None

    # Loading the data
    Data = load_file(main_data)
    if main_data_y: # N x m main_data and separate y data. main_data may be 3D for an RNN model
        X_original = Data
        y_original = load_file(main_data_y)
    elif len(Data.shape) == 2: # Typical case with an N x (m+1) main_data
        X_original = Data[:, :-1]
        y_original = Data[:, -1]
    if len(y_original.shape) == 1: # Ensuring y_original is 2D and in an N_samples x N_features form
        y_original = y_original[:, np.newaxis]

    if test_data:
        Test_data = load_file(test_data)
        if test_data_y: # N x m test_data and separate y data. test_data may be 3D for an RNN model
            X_test_original = Test_data
            y_test_original = load_file(test_data_y)
        else: # Typical case with an N x (m+1) test_data
            X_test_original = Test_data[:, :-1]
            y_test_original = Test_data[:, -1]
        if len(y_test_original.shape) == 1: # Ensuring y_test_original is 2D and in an N_samples x N_features form
            y_test_original = y_test_original[:, np.newaxis]
    else:
        X_test_original = None
        y_test_original = None

    if cat is None:
        cat = [0] * X_original.shape[1]
    # Ensuring the user didn't pass too many plot labels by mistake
    if isinstance(xticks, (list, tuple)):
        xticks = xticks[:X_original.shape[1]]
    if isinstance(yticks, (list, tuple)):
        yticks = yticks[:y_original.shape[1]]
    if classification and class_weight is None:
        class_weight = np.ones(len( set(y_original.squeeze()) ))

    # Selecting a model
    if model_name is None:
        # Determining nonlinearity and multicollinearity automatically
        nonlinear = nonlinearity_assess(X_original, y_original, plot_interrogation, cat, significance, difference = 0.4, xticks = xticks, yticks = yticks)
        multicollinear = collinearity_assess(X_original, y_original, plot_interrogation, xticks, yticks)
        if not nonlinear and dynamic_model:
            nonlinear = nonlinearity_assess_dynamic(X_original, y_original, plot_interrogation, alpha = significance, difference = 0.4, xticks = xticks, yticks = yticks, lag = max(lag))
        # Automatically selecting a model based on the nonlinear and multicollinear test results
        if nonlinear:
            model_name = ['LCEN'] # Used no matter whether the system is dynamic or not, as the lag hyperparameter takes care of the dynamics
            # Nonlinear, nondynamic models
            if not dynamic_model:
                lag, min_lag = [0], 0
                if interpretable:
                    if verbosity_level: print('As your data are nonlinear and you require an interpretable model, only LCEN will be used.')
                elif continuity:
                    if verbosity_level: print('As your data are nonlinear, you do not require the model to be interpretable, and you require continuity, LCEN, SVM, and MLP will be tested')
                    model_name.append('SVM')
                    model_name.append('MLP')
                else:
                    if verbosity_level: print('As your data are nonlinear, you do not require the model to be interpretable, and you do not require continuity, LCEN, SVM, MLP, RF, and AdaBoost will be tested')
                    model_name.append('SVM')
                    model_name.append('MLP')
                    model_name.append('RF')
                    model_name.append('AdaB')
            # Nonlinear, dynamic models
            else:
                if lag == [0]:
                    lag = list(range(1+min_lag, 6+min_lag))
                if interpretable:
                    if verbosity_level: print(f'As your data are nonlinear and you require an interpretable model, LCEN with lag = {lag} will be used.')
                else:
                    if verbosity_level: print(f'As your data are nonlinear and you do not require the model to be interpretable, LCEN with lag = {lag} and an RNN will be used.')
                    model_name.append('RNN')
        # Linear, nondynamic models
        elif not dynamic_model:
            lag, min_lag = [0], 0
            degree = [1]
            if not multicollinear:
                if verbosity_level: print('As there is no significant nonlinearity and multicollinearity in the data, OLS will be used.')
                model_name = ['OLS']
            elif interpretable:
                if verbosity_level: print('As you require an interpretable model, EN, SPLS, and LCEN with degree = 1 will be used.')
                model_name = ['EN', 'SPLS', 'LCEN']
            else:
                if verbosity_level: print('As your data have significant multicollinearity and you do not require an interpretable model, EN, SPLS, PLS, and LCEN with degree = 1 will be used.')
                model_name = ['EN', 'SPLS', 'PLS', 'LCEN']
        # Linear dynamic models
        else:
            if lag == [0]:
                lag = list(range(1+min_lag, 6+min_lag))
            degree = [1]
            trans_type = 'poly'
            if verbosity_level: print(f'As your data have significant dynamics but are linear, LCEN with degree = 1 and lag = {lag} will be used.')
            model_name = ['LCEN']

    # Cross-Validation Strategy
    if cv_method is None:
        if not dynamic_model and group_name is None:
            cv_method = 'Re_KFold'
            if verbosity_level: print(f'{"Nested "*nested_cv}CV with repeated KFold in inner loop {"and one-std rule "*robust_priority}will be used.')
        elif not dynamic_model:
            cv_method = 'GroupKFold'
            if verbosity_level: print(f'{"Nested "*nested_cv}GroupKFold {"with one-std rule "*robust_priority}will be used.')
        # Dynamic models
        elif nested_cv:
            cv_method = 'Timeseries'
            if verbosity_level: print(f'Cross-validation for time series {"with one-std rule "*robust_priority}will be used.')
        elif robust_priority:
            cv_method = 'BIC'
            if verbosity_level: print('BIC (Bayesian information criterion) will be used.')
        else:
            cv_method = 'AICc'
            if verbosity_level: print('AICc (Akaike information criterion with correction) will be used.')

    # Preprocessing the data
    X = X_original.copy('F') # Switching to Fortran order to speed up Elastic Net (and LCEN) [but does not make any significant difference]
    y = y_original.copy('F')
    # Scaling the data
    if scale_X and len(X.shape) == 2 and X.shape[1] > 0: # StandardScaler doesn't work with 3D arrays or with arrays that are 2D but empty in one dimension
        scaler_x = StandardScaler(with_mean=True, with_std=True)
        scaler_x.fit(X)
        X_scale = scaler_x.transform(X)
    else:
        X_scale = X
    if scale_y and not classification:
        scaler_y = StandardScaler(with_mean=True, with_std=True)
        scaler_y.fit(y)
        y_scale = scaler_y.transform(y)
    elif classification:
        scale_y = False # If doing classification, y has class labels, and thus should not be scaled
        y = np.array(y.squeeze(), dtype = int)
        y_scale = np.array(y.squeeze(), dtype = int)
    else:
        y_scale = y

    if X_test_original is not None:
        X_test = X_test_original.copy('F')
        y_test = y_test_original.copy('F')
        if scale_X and len(X.shape) == 2 and X.shape[1] > 0: # StandardScaler doesn't work with 3D arrays
            X_test_scale = scaler_x.transform(X_test)
        else:
            X_test_scale = X_test
        if scale_y and not classification:
            y_test_scale = scaler_y.transform(y_test)
        elif classification:
            y_test = np.array(y_test.squeeze(), dtype = int)
            y_test_scale = np.array(y_test.squeeze(), dtype = int)
        else:
            y_test_scale = y_test
    else:
        X_test = X
        y_test = y
        X_test_scale = X_scale
        y_test_scale = y_scale

    # Setting up a dict to save results (hyperparameters and predictions)
    fitting_result = {}
    general_hyper = {'SPA version': '1.5.0', 'CV method': cv_method, 'Number of folds (K)': K_fold} # Some of some model-independent hyperparameters ### TODO: properly get SPA.__version__
    if 're' in cv_method.casefold():
        general_hyper['Number of CV repeats'] = Nr
    general_hyper['Robust priority'] = robust_priority
    general_hyper['Train data'] = f'File named \'{main_data}\' with shape {X.shape}'
    if main_data_y:
        general_hyper['Train data, y'] = f'File named \'{main_data_y}\''
    if test_data:
        general_hyper['Test data'] = f'File named \'{test_data}\' with shape {X_test.shape}'
    if test_data_y:
        general_hyper['Test data, y'] = f'File named \'{test_data_y}\''
    # Model fitting
    if 'OLS' in model_name:
        from regression_models import OLS_fitting
        final_model, model_params, mse_train, mse_test, yhat_train, yhat_test = OLS_fitting(X_scale, y_scale, X_test_scale, y_test_scale, classification, class_weight)
        fitting_result['OLS'] = OrderedDict({'final_model': final_model, 'model_params': model_params, 'mse_train': mse_train, 'mse_test': mse_test, 'yhat_train': yhat_train, 'yhat_test': yhat_test})
    else: # TODO: how do we compare OLS with the other models if OLS doesn't have validation scores?
        if not nested_cv: # Static / traditional CV
            for index, this_model in enumerate(model_name):
                if this_model in {'LCEN', 'SVM', 'RF', 'GBDT', 'AdaB', 'EN', 'PLS', 'SPLS'}: # There may be other models if the user passed model_name manually
                    if verbosity_level >= 2: print(f'Running model {this_model}', end = '\r')
                    fitting_result[this_model], _ = run_cv_ML(this_model, X, y, X_scale, y_scale, X_test, y_test, X_test_scale, y_test_scale, cv_method, group,
                                                K_fold, Nr, scale_X, scale_y, classification, l1_ratio, alpha, SPLS_K, SPLS_eta, lag, min_lag, robust_priority, degree,
                                                trans_type, LCEN_cutoff, LCEN_interaction, LCEN_transform_y, RF_n_estimators, RF_max_depth, RF_min_samples_leaf, RF_n_features,
                                                learning_rate, SVM_gamma, SVM_C, SVM_epsilon, verbosity_level)
                    if verbosity_level: print(f'Completed model {this_model}')
                elif this_model in {'MLP', 'RNN'}: # There may be other models if the user passed model_name manually
                    temp = cv.CV_mse(this_model, X_scale, y_scale, X_test_scale, y_test_scale, X, y, cv_type = cv_method, group = group, K_fold = K_fold, Nr = Nr,
                        scale_X = scale_X, scale_y = scale_y, classification = classification, activation = activation, MLP_layers = MLP_layers, RNN_layers = RNN_layers,
                        batch_size = batch_size, learning_rate = learning_rate, weight_decay = weight_decay, l1_penalty_factor = l1_penalty_factor, n_epochs = n_epochs,
                        class_weight = class_weight, scheduler = scheduler, scheduler_mode = scheduler_mode, scheduler_factor = scheduler_factor,
                        scheduler_patience = scheduler_patience, scheduler_last_epoch = scheduler_last_epoch, scheduler_warmup = scheduler_warmup,
                        val_loss_file = val_loss_file, expand_hyperparameter_search = expand_hyperparameter_search, verbosity_level = verbosity_level)
                    fitting_result[this_model] = OrderedDict({'final_model': temp[0], 'mse_train': temp[2], 'mse_val': temp[1].min().min(), 'mse_test': temp[3],
                                                        'best_hyperparameters': temp[6], 'yhat_train': temp[4], 'yhat_test': temp[5]})
                    if verbosity_level: print(f'Completed model {this_model}' + ' '*15)

        else: # Nested CV
            if group_name is None:
                from sklearn.model_selection import train_test_split
                MSE_val = np.empty((len(model_name), num_outer)) * np.nan
                for index_out in range(num_outer):
                    if verbosity_level >= 3: print(f'Beginning nested CV loop {index_out+1} out of {num_outer}', end = '\r')
                    X_nest, X_nest_val, y_nest, y_nest_val = train_test_split(X, y, test_size = 1/K_fold, random_state = index_out)
                    X_nest_scale, X_nest_scale_val, y_nest_scale, y_nest_scale_val = train_test_split(X_scale, y_scale, test_size = 1/K_fold, random_state = index_out)
                    for index, this_model in enumerate(model_name):
                        if this_model in {'LCEN', 'SVM', 'RF', 'GBDT', 'AdaB', 'EN', 'PLS', 'SPLS'}: # There may be other models if the user passed model_name manually
                            MSE_val[index, index_out] = run_cv_ML(this_model, X_nest, y_nest, X_nest_scale, y_nest_scale, X_nest_val, y_nest_val, X_nest_scale_val, y_nest_scale_val,
                                    cv_method, group, K_fold, Nr, scale_X, scale_y, l1_ratio, alpha, SPLS_K, SPLS_eta, lag, min_lag, robust_priority, degree, trans_type,
                                    LCEN_cutoff, LCEN_transform_y, LCEN_interaction, RF_n_estimators, RF_max_depth, RF_min_samples_leaf, RF_n_features, learning_rate, SVM_gamma,
                                    SVM_C, SVM_epsilon, verbosity_level, True)
            else:
                from sklearn.model_selection import LeaveOneGroupOut
                MSE_val = np.empty((len(model_name), len(np.unique(group)))) * np.nan
                RMSE_val = np.empty((len(model_name), len(np.unique(group)))) * np.nan
                logo = LeaveOneGroupOut()
                for index_out, (train, val) in enumerate( logo.split(X, y.flatten(), groups = group.flatten()) ):
                    if verbosity_level >= 3: print(f'Beginning nested CV loop {index_out+1} out of {len( set(group.flatten()) )}', end = '\r')
                    for index, this_model in enumerate(model_name):
                        if this_model in {'LCEN', 'SVM', 'RF', 'GBDT', 'AdaB', 'EN', 'PLS', 'SPLS'}: # There may be other models if the user passed model_name manually
                            MSE_val[index, index_out] = run_cv_ML(this_model, X[train], y[train], X_scale[train], y_scale[train], X[val], y[val], X_scale[val], y_scale[val],
                                    cv_method, group[train], K_fold, Nr, scale_X, scale_y, l1_ratio, alpha, SPLS_K, SPLS_eta, lag, min_lag, robust_priority, degree,
                                    trans_type, LCEN_cutoff, LCEN_interaction, LCEN_transform_y, RF_n_estimators, RF_max_depth, RF_min_samples_leaf, RF_n_features,
                                    learning_rate, SVM_gamma, SVM_C, SVM_epsilon, verbosity_level, True)

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
            fitting_result[local_selected_model], _ = run_cv_ML(local_selected_model, X, y, X_scale, y_scale, X_test, y_test, X_test_scale, y_test_scale,
                    cv_method, group, K_fold, Nr, scale_X, scale_y, l1_ratio, alpha, SPLS_K, SPLS_eta, lag, min_lag, robust_priority, degree, trans_type,
                    LCEN_cutoff, LCEN_interaction, LCEN_transform_y, RF_n_estimators, RF_max_depth, RF_min_samples_leaf, RF_n_features, learning_rate,
                    SVM_gamma, SVM_C, SVM_epsilon, verbosity_level)

    # Finding the best model
    for idx, entry in enumerate(fitting_result): # TODO: this will probably not work with OLS, since it doesn't have a mse_val entry (see above)
        if idx == 0 or fitting_result[entry]['mse_val'] < fitting_result[selected_model]['mse_val']:
            selected_model = entry
    # Catching wrong model names
    for model in model_name:
        if model not in fitting_result:
            warnings.warn(f'{model} is not a valid model name, so it was ignored.')
    if 'selected_model' not in locals():
        raise UnboundLocalError(f'You input {model_name} for model_name, but that is not a valid name.')

    # Formatting a dictionary for the output
    fitting_result[selected_model]['general_hyper'] = general_hyper
    fitting_result[selected_model].move_to_end('general_hyper', last = False) # Move to beginning
    if 'model_hyper' in fitting_result[selected_model] and 'lag' in fitting_result[selected_model]['model_hyper'].keys(): # The first "lag" entries are removed from yhat, so we need to remove them from X and y
        lag = fitting_result[selected_model]['model_hyper']['lag']
    else:
        lag = 0
    if classification:
        fitting_result[model]['logits_train_normalized'] = (fitting_result[selected_model]['yhat_train'].T / fitting_result[selected_model]['yhat_train'].sum(axis=1)).T
        fitting_result[model]['predicted_class_train'] = fitting_result[model]['logits_train_normalized'].argmax(axis = 1) # TODO: for binary classification, allow thresholds
        fitting_result[model]['logits_test_normalized'] = (fitting_result[selected_model]['yhat_test'].T / fitting_result[selected_model]['yhat_test'].sum(axis=1)).T
        fitting_result[model]['predicted_class_test'] = fitting_result[model]['logits_test_normalized'].argmax(axis = 1)
        _ = fitting_result[model].pop('yhat_train')
        _ = fitting_result[model].pop('yhat_test')
    else:
        if scale_y and selected_model != 'LCEN': # LCEN already returns unscaled predictions
            fitting_result[selected_model]['yhat_train_nontrans'] = scaler_y.inverse_transform(np.atleast_2d(fitting_result[selected_model]['yhat_train']))
            fitting_result[selected_model]['yhat_test_nontrans'] = scaler_y.inverse_transform(np.atleast_2d(fitting_result[selected_model]['yhat_test']))
        else:
            fitting_result[selected_model]['yhat_train_nontrans'] = fitting_result[selected_model]['yhat_train']
            fitting_result[selected_model]['yhat_test_nontrans'] = fitting_result[selected_model]['yhat_test']
            del fitting_result[selected_model]['yhat_train']
            del fitting_result[selected_model]['yhat_test']
        fitting_result[selected_model]['yhat_train_nontrans_mean'] = np.mean(fitting_result[selected_model]['yhat_train_nontrans'])
        fitting_result[selected_model]['yhat_train_nontrans_stdev'] = np.std(fitting_result[selected_model]['yhat_train_nontrans'])
        fitting_result[selected_model]['MSE_train_nontrans'] = np.mean( (fitting_result[selected_model]['yhat_train_nontrans'].squeeze() - y[lag:].squeeze())**2 )
        fitting_result[selected_model]['RMSE_train_nontrans'] = np.sqrt(fitting_result[selected_model]['MSE_train_nontrans'])
        SSErr = ((fitting_result[selected_model]['yhat_train_nontrans'].squeeze() - y[lag:].squeeze())**2).sum()
        SST = ((y[lag:].squeeze() - y[lag:].squeeze().mean())**2).sum()
        fitting_result[selected_model]['R^2_train'] = 1 - SSErr/SST
        non_zero = y[lag:].squeeze() != 0 # Avoiding infinity mean relative errors if at least one entry in y is 0
        fitting_result[selected_model]['Mean_relative_error_train'] = np.nanmean( np.abs((fitting_result[selected_model]['yhat_train_nontrans'].squeeze()[non_zero] - y[lag:].squeeze()[non_zero]) / y[lag:].squeeze()[non_zero]) )
        # If no test set was used, these test set statistics are equal to those from the train set
        if fitting_result[selected_model]['yhat_train_nontrans'].shape != fitting_result[selected_model]['yhat_test_nontrans'].shape or not(np.all(fitting_result[selected_model]['yhat_train_nontrans'] == fitting_result[selected_model]['yhat_test_nontrans'])):
            fitting_result[selected_model]['yhat_test_nontrans_mean'] = np.mean(fitting_result[selected_model]['yhat_test_nontrans'])
            fitting_result[selected_model]['yhat_test_nontrans_stdev'] = np.std(fitting_result[selected_model]['yhat_test_nontrans'])
            fitting_result[selected_model]['MSE_test_nontrans'] = np.mean( (fitting_result[selected_model]['yhat_test_nontrans'].squeeze() - y_test.squeeze())**2 )
            fitting_result[selected_model]['RMSE_test_nontrans'] = np.sqrt(fitting_result[selected_model]['MSE_test_nontrans'])
            SSErr = ((fitting_result[selected_model]['yhat_test_nontrans'].squeeze() - y_test.squeeze())**2).sum()
            SST = ((y_test.squeeze() - y_test.squeeze().mean())**2).sum()
            fitting_result[selected_model]['R^2_test'] = 1 - SSErr/SST
            non_zero = y_test.squeeze() != 0 # Avoiding infinity mean relative errors if at least one entry in y_test is 0
            fitting_result[selected_model]['Mean_relative_error_test'] = np.mean( np.abs((fitting_result[selected_model]['yhat_test_nontrans'].squeeze()[non_zero] - y_test.squeeze()[non_zero]) / y_test.squeeze()[non_zero]) )
    # Residual analysis + test for dynamics in the residual
    if len(y_test.squeeze()) >= 4 and not classification and (selected_model not in {'RNN', 'LCEN'} or (selected_model == 'LCEN' and lag == 0)): # TODO: residuals with small lengths lead to errors when plotting ACF. Need to figure out why
        _, dynamic_test_result = residual_analysis(X_test, y_test.squeeze(), fitting_result[selected_model]['yhat_test_nontrans'].squeeze(), plot_interrogation, alpha = significance, round_number = selected_model)
        if dynamic_test_result and not(dynamic_model) and (selected_model not in {'RNN', 'LCEN'} or (selected_model == 'LCEN' and lag == 0)) and verbosity_level:
            print('A residual analysis found dynamics in the system. Please run SPA again with dynamic_model = True or specify a dynamic model via model_name.')
            print('Note that specifying any model via model_name (including nondynamic ones) will override the dynamic_model flag.')

    # Setup for saving
    # jsons do not work with numpy arrays - converting to list
    fr2 = deepcopy(fitting_result)
    for model in fr2.keys():
        del fr2[model]['final_model'] # Models aren't convertible to json
        for top_key, top_value in fr2[model].items():
            if isinstance(fr2[model][top_key], dict):
                for key, value in fr2[model][top_key].items():
                    if isinstance(value, np.ndarray):
                        fr2[model][top_key][key] = value.tolist()
            elif isinstance(top_value, np.ndarray):
                fr2[model][top_key] = top_value.squeeze().tolist()
    if 'time_now' not in locals(): # Nested validation already creates this variable, so we won't create it again to keep things consistent
        time_now = '-'.join([str(elem) for elem in localtime()[:6]]) # YYYY-MM-DD-hh-mm-ss
    if selected_model in {'MLP', 'RNN'}:
        torchsave(fitting_result[selected_model]['final_model'].state_dict(), f'SPA_{selected_model}-model_{time_now}.pt')
    # Saving as a pickled file
    with open(f'SPA_results_{time_now}.p', 'wb') as f:
        pickle.dump(fitting_result, f)
    # Saving as a json file
    with open(f'SPA_results_{time_now}.json', 'w') as f:
        # Numpy scalars (e.g. PLS n_components as int64) are not caught by the ndarray
        # conversion above and are not JSON-serializable; convert them on the fly.
        json.dump(fr2, f, indent = 4, default = lambda o: o.item() if isinstance(o, np.generic) else str(o))
    if verbosity_level: print(f'The best model is {selected_model}. View its results via fitting_result["{selected_model}"] or by opening the SPA_results json/pickle files.')
    if verbosity_level >= 2 and not classification: print(f'Train set: RMSE = {fitting_result[selected_model]["RMSE_train_nontrans"]:.4f} | Mean relative error = {fitting_result[selected_model]["Mean_relative_error_train"]:.4f}')
    if verbosity_level and not classification: print(f'Test set : RMSE = {fitting_result[selected_model]["RMSE_test_nontrans"]:.4f} | Mean relative error = {fitting_result[selected_model]["Mean_relative_error_test"]:.4f}')
    if verbosity_level >= 2 and classification: print(f'Train set: Mean F1 = {fitting_result[selected_model]["mse_train"]["Mean F1"]:.4f} | MCC = {fitting_result[selected_model]["mse_train"]["MCC"]:.4f}')
    if verbosity_level and classification: print(f'Test set : Mean F1 = {fitting_result[selected_model]["mse_test"]["Mean F1"]:.4f} | MCC = {fitting_result[selected_model]["mse_test"]["MCC"]:.4f}')
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
    elif ext == '.npy':
        my_file = np.load(filename)
    else:
        raise ValueError(f'Please provide a filename with extension in {{.txt, .csv, .tsv, .xls, .xlsx, .npy}}. You passed {filename}')
    return my_file

def run_cv_ML(model_index, X_train, y_train, X_train_scaled, y_train_scaled, X_test, y_test, X_test_scaled, y_test_scaled, cv_method, group, K_fold, Nr, scale_X, scale_y,
              classification, l1_ratio, alpha, SPLS_K, SPLS_eta, lag, min_lag, robust_priority, degree, trans_type, LCEN_cutoff, LCEN_interaction, LCEN_transform_y, RF_n_estimators,
              RF_max_depth, RF_min_samples_leaf, RF_n_features, learning_rate, SVM_gamma, SVM_C, SVM_epsilon, verbosity_level, for_nested_validation = False):
    """
    Runs a nondynamic model for CV or final-run purposes. Automatically called by SPA.

    Parameters
    ----------
    model_index to verbosity_level
        Automatically called by SPA based on what was passed to main_SPA()
    for_nested_validation : bool, optional, default = False
        Whether the run is done to validate a model through nested validation (NV) (to determine ...
            the best hyperparameters) or test the best model obtained from CV/NV.
        This changes a little the syntax and values returned, but not the logic.
    """
    if for_nested_validation:
        # For the sake of clarity
        X_val, y_val = X_test, y_test
        X_val_scaled, y_val_scaled = X_test_scaled, y_test_scaled

        if model_index == 'LCEN':
            _, _, _, _, mse_val, _, _, _ = cv.CV_mse(model_index, X_train, y_train, X_val, y_val, None, None, cv_method, K_fold, Nr, scale_X = scale_X, scale_y = scale_y,
                    group = group, classification = classification, alpha = alpha, l1_ratio = l1_ratio, lag = lag, min_lag = min_lag, label_name = True, robust_priority = robust_priority,
                    degree = degree, trans_type = trans_type, LCEN_cutoff = LCEN_cutoff, LCEN_interaction = LCEN_interaction, LCEN_transform_y = LCEN_transform_y, verbosity_level = verbosity_level)
        elif model_index in {'RF', 'GBDT', 'AdaB'}:
            _, _, _, mse_val, _, _, _ = cv.CV_mse(model_index, X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, X_train, y_train, cv_method, K_fold, Nr,
                    scale_X = scale_X, scale_y = scale_y, group = group, classification = classification, robust_priority = robust_priority, RF_n_estimators = RF_n_estimators,
                    RF_max_depth = RF_max_depth, RF_min_samples_leaf = RF_min_samples_leaf, RF_n_features = RF_n_features, learning_rate = learning_rate, verbosity_level = verbosity_level)
        elif model_index == 'SVM':
            _, _, _, mse_val, _, _, _ = cv.CV_mse(model_index, X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, X_train, y_train, cv_method, K_fold, Nr, scale_X = scale_X,
                    scale_y = scale_y, group = group, classification = classification, robust_priority = robust_priority, SVM_gamma = SVM_gamma, SVM_C = SVM_C, SVM_epsilon = SVM_epsilon,
                    verbosity_level = verbosity_level)
        else: # EN, PLS, and SPLS
            _, _, _, _, mse_val, _, _, _ = cv.CV_mse(model_index, X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, X_train, y_train, cv_method, K_fold, Nr, scale_X = scale_X,
                    scale_y = scale_y, group = group, classification = classification, alpha = alpha, l1_ratio = l1_ratio, SPLS_K = SPLS_K, SPLS_eta = SPLS_eta, robust_priority = robust_priority,
                    verbosity_level = verbosity_level)
        return mse_val
    else:
        if model_index == 'LCEN':
            model_hyper, final_model, model_params, mse_train, mse_test, yhat_train, yhat_test, mse_val = cv.CV_mse(model_index, X_train, y_train, X_test, y_test, None, None,
                    cv_method, K_fold, Nr, scale_X = scale_X, scale_y = scale_y, group = group, classification = classification, alpha = alpha, l1_ratio = l1_ratio, lag = lag,
                    min_lag = min_lag, label_name = True, robust_priority = robust_priority, degree = degree, trans_type = trans_type, LCEN_cutoff = LCEN_cutoff,
                    LCEN_interaction = LCEN_interaction, LCEN_transform_y = LCEN_transform_y, verbosity_level = verbosity_level)
            fitting_result = {'model_hyper': model_hyper, 'final_model': final_model, 'model_params': model_params, 'mse_train': mse_train, 'mse_val': mse_val,
                              'mse_test': mse_test, 'yhat_train': yhat_train, 'yhat_test': yhat_test}
        elif model_index in {'RF', 'GBDT', 'AdaB'}:
            model_hyper, final_model, mse_train, mse_test, yhat_train, yhat_test, mse_val = cv.CV_mse(model_index, X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled,
                    X_train, y_train, cv_method, K_fold, Nr, scale_X = scale_X, scale_y = scale_y, group = group, classification = classification, robust_priority = robust_priority,
                    RF_n_estimators = RF_n_estimators, RF_max_depth = RF_max_depth, RF_min_samples_leaf = RF_min_samples_leaf, RF_n_features = RF_n_features, learning_rate = learning_rate,
                    verbosity_level = verbosity_level)
            fitting_result = {'model_hyper': model_hyper, 'final_model': final_model, 'mse_train': mse_train, 'mse_val': mse_val, 'mse_test': mse_test,
                              'yhat_train': yhat_train, 'yhat_test': yhat_test}
        elif model_index == 'SVM':
            model_hyper, final_model, mse_train, mse_test, yhat_train, yhat_test, mse_val = cv.CV_mse(model_index, X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled,
                    X_train, y_train, cv_method, K_fold, Nr, scale_X = scale_X, scale_y = scale_y, group = group, classification = classification, robust_priority = robust_priority,
                    SVM_gamma = SVM_gamma, SVM_C = SVM_C, SVM_epsilon = SVM_epsilon, verbosity_level = verbosity_level)
            fitting_result = {'model_hyper': model_hyper, 'final_model': final_model, 'mse_train': mse_train, 'mse_val': mse_val, 'mse_test': mse_test,
                              'yhat_train': yhat_train, 'yhat_test': yhat_test}
        else: # EN, PLS, and SPLS
            model_hyper, final_model, model_params, mse_train, mse_test, yhat_train, yhat_test, mse_val = cv.CV_mse(model_index, X_train_scaled, y_train_scaled, X_test_scaled,
                    y_test_scaled, X_train, y_train, cv_method, K_fold, Nr, scale_X = scale_X, scale_y = scale_y, group = group, classification = classification, alpha = alpha,
                    l1_ratio = l1_ratio, SPLS_K = SPLS_K, SPLS_eta = SPLS_eta, robust_priority = robust_priority, verbosity_level = verbosity_level)
            fitting_result = {'model_hyper': model_hyper, 'final_model': final_model, 'model_params': model_params, 'mse_train': mse_train, 'mse_val': mse_val,
                              'mse_test': mse_test, 'yhat_train': yhat_train, 'yhat_test': yhat_test}
        return OrderedDict(fitting_result), mse_val
