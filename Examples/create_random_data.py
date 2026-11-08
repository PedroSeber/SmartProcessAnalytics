import numpy as np
import pandas as pd
from time import localtime

def create_random_data(data_size, mode, data_range = (-10, 10), degrees = [1], seed = 123456789, coefficients = None, noise = (0, 0), false_vars = 0):
    """
    Creates "data_size" random points with values from data_range[0] to data_range[1], then use them and a functional form ...
        (defined by "mode" and "degrees") to generate a y response.
    """
    # Creating the X_data for the dataset
    rng = np.random.default_rng(seed)
    if mode.casefold() in {'multicollinear', 'multicollinearity'}:
        X_0 = rng.integers(data_range[0]*10, data_range[1]*10, (data_size[0], 1)) / 10
        X_noise_sigma = 0.05 # Edit as needed. New variable so that this can be added to the file_name below
        noise_MC = rng.normal(0, X_noise_sigma, X_0.shape)
        X_1 = X_0 + noise_MC
        X_data = np.concatenate((X_0, X_1), axis = 1)
        coefficients = np.array([4, 2])
        print(f'The X_1 noise level for the multicollinear model is {np.mean(np.abs(noise_MC/X_0))*100:.1f}%')
    else: # General case
        X_data = rng.integers(data_range[0]*10, data_range[1]*10, data_size) / 10 # Numbers between data_range[0] and data_range[1] with one decimal place
    # Setting up the other parameters for the function
    noise = tuple(int(elem) if elem == int(elem) else elem for elem in noise) # Changes integer noise parameters to actual ints for display convenience
    noise_vals = rng.normal(noise[0], noise[1], X_data.shape[0]) # Always run to "burn" X_data.shape[0] values from rng to ensure reproducibility between noisy and noiseless runs
    if noise[1]:
        noise_str = f' + ϵ{noise}'
    else:
        noise_str = ''
        noise_vals = 0
    if coefficients is None:
        coefficients = rng.integers(-100, 100, X_data.shape[1]) / 10
        coefficients[coefficients == 0] = 0.1 # Avoiding coefficients = 0
    else:
        _ = rng.integers(-100, 100, X_data.shape[1]) / 10 # Ran to "burn" X_data.shape[1] values from the rng to ensure reproducibility
    degrees = [int(elem) if elem == int(elem) else elem for elem in degrees] # Changes integer degrees to actual ints for display convenience
    powers = rng.choice(degrees, X_data.shape[1])
    # Creating the y_data based on the input parameters
    if mode.casefold() == 'poly':
        is_ln = [False]*X_data.shape[1]
        y_data = (coefficients * X_data**powers).sum(axis = 1)
    elif mode.casefold() == 'log':
        is_ln = [True]*X_data.shape[1]
        y_data = (coefficients * np.log(np.abs(X_data))**powers).sum(axis = 1)
    elif mode.casefold() in {'stefan-boltzmann', 'stefanboltzmann', 'stefan_boltzmann'}:
        mode = 'stefanboltzmann' # For standardization
        degrees = [1] # For standardization since we have a fixed eqn
        y_data = 3.828e26 * X_data[:, 0]**2 * X_data[:, 1]**4 # L_{Sun} * R^2 * T^4 [In units of R_{Sun} and T_{Sun}]
        functional_form_str = f'L = 3.828e26 * R^2 * T^4{noise_str}'
    elif mode.casefold() == 'energy': # NOTE: if irrelevant variables are not needed, set data_size to (K, 1), as V always gets added later
        degrees = [1] # For standardization since we have a fixed eqn, even though there's a degree 4 interaction with m^2*v^2
        V = rng.integers(5e8, 2.5e9, data_size) / 10 # Numbers between 50 000 000 (5e7) and 250 000 000 (2.5e8) with one decimal place
        X_data = np.hstack((X_data, V))
        c = 299792458 # m / s
        y_data = c**2 * X_data[:, 0]**2 * X_data[:, 1]**2 + c**4 * X_data[:, 0]**2 # E^2
        functional_form_str = f'E^2 = 8.98755e16*m^2*v^2 + 8.07761e33*m^2{noise_str}'
        file_name = f'{mode}_{data_size[0]}x{data_size[1]}-data_{data_range[0]}to{data_range[1]}-range_{seed}-seed_({noise[0]},{noise[1]:.2e})-noise.csv'
        print(f'Relativistic effects account for {np.mean(1 - c**4 * X_data[:, 0]**2/y_data)*100:.2f}% of this system\'s energy')
    elif mode.casefold() in {'multicollinear', 'multicollinearity'}:
        mode = 'multicollinear'
        y_data = (coefficients*X_data).sum(axis = 1)
        functional_form_str = f'y = 4*X_0 + 2*X_1{noise_str}'
        file_name = f'{mode}_{data_size[0]}x2-data_{data_range[0]}to{data_range[1]}-range_{seed}-seed_({noise[0]},{noise[1]})-ynoise_(0,{X_noise_sigma})-Xnoise.csv'
    else: # TODO: mixture of log and linear, maybe other functional forms
        raise NotImplementedError
    # Adding false variables that have no effect on the y values
    X_false = rng.integers(data_range[0]*10, data_range[1]*10, (data_size[0], false_vars)) / 10 # Numbers between data_range[0] and data_range[1] with one decimal place
    # Saving the data using SPA's preferred format
    y_data_noisy = y_data + noise_vals
    all_data = np.column_stack((X_data, X_false, y_data_noisy))
    degree_str = str(degrees)[1:-1]
    coeff_str = str(coefficients)[1:-1].replace(",", "")
    if 'file_name' not in locals(): # Specific equations already have the file_name built-in within their if-statement
        file_name = f'{mode}_{data_size[0]}x{data_size[1]}-data_{data_range[0]}to{data_range[1]}-range_{degree_str.replace(" ", "")}-degree_{seed}-seed_({noise[0]},{noise[1]})-noise_{false_vars:04}-false-vars.csv'
    np.savetxt(file_name, all_data, delimiter = ',', comments = '', fmt='%.8g')
    # Printing the final functional form
    if 'functional_form_str' not in locals(): # Specific equations already have the functional_form_str built-in within their if-statement
        functional_form_str = [f'{coefficients[idx]}*ln(x{idx})^{powers[idx]}' if is_ln[idx] else f'{coefficients[idx]}*(x{idx})^{powers[idx]}' for idx in range(X_data.shape[1])] # coefficients*log(x)^power or coeffiicents*x^power
        functional_form_str = ' + '.join([elem[:-2] + elem[-2:].replace('^1', '') for elem in functional_form_str]) # Removing ^1. String addition to avoid removing ^10, ^11, ^12, ...
        functional_form_str = f'y = {functional_form_str}{noise_str}'.replace('+ -', '- ')
    if X_data.shape[1] <= 25:
        print('The final functional form for this dataset is:')
        print(functional_form_str)
    print(f'The noise level is {np.mean(np.abs(noise_vals/y_data))*100:.1f}%')
    # Writing a log file
    time_now = '-'.join([str(elem) for elem in localtime()[:6]]) # YYYY-MM-DD-hh-mm-ss
    log_text = f'Log for create_random_data()\n' + (
               f'Function call: python create_random_data.py {data_size[0]} {data_size[1]} {mode} -dr {data_range[0]} {data_range[1]} -d {degree_str.replace(",", "")} -s {seed} -c {coeff_str} -n {noise[0]} {noise[1]} -fv {false_vars}\n') + (
               f'Functional form: {functional_form_str}\n') + (
               f'Noise level: {np.mean(np.abs(noise_vals/y_data))*100:.1f}%\n') + (
               f'False input variables: {false_vars}\n') + (
               f'Completion time: {time_now}\n')
    with open(file_name[:-3] + 'log', 'w') as f:
        f.write(log_text)

if __name__ == '__main__':
    # Input setup
    import argparse
    parser = argparse.ArgumentParser(description = 'Creates "data_size" random points, then use them and a functional form (defined by "mode" and "degrees") to generate a y response.')
    parser.add_argument('data_size', type = int, nargs = 2, help = 'The size of the X data that will be randomly created.')
    parser.add_argument('mode', type = str, nargs = 1, help = 'The functional form of the dataset response. Must be in {"poly", "log", "energy", "multicollinear"}. "Muticollinear" is for a specific test, and in general should not be called by the user.')
    parser.add_argument('-dr', '--datarange', type = int, nargs = 2, metavar = (-10, 10), default = (-10, 10), help = 'Integers representing the minimum and maximum values of the X data. ' +
                        'The limits are multiplied by 10, used to generate random integers, then divided by 10 to turn the X data into floats with one decimal digit.')
    parser.add_argument('-d', '--degrees', type = float, nargs = '+', metavar = 1, default = [1], help = 'A list of valid degrees to which the X data is randomly raised.' +
                        'For example, degrees = [1, 2] will include linear and quadratic terms at random, degrees = [2] will include only quadratic terms.')
    parser.add_argument('-s', '--seed', type = int, nargs = 1, metavar = 'int', default = [123456789],
        help = 'The seed that is passed to np.random.default_rng() for reproducibility.')
    parser.add_argument('-c', '--coefficients', type = float, nargs = '+', metavar = 'None or list', default = None,
        help='The coefficients that multiply each feature of the X data. If None, they are generated at random between -10 and 10.')
    parser.add_argument('-n', '--noise', type = float, nargs = 2, metavar = (0, 0), default = (0, 0), help='The mean and stdev of the Gaussian noise added to the y data. By default, no noise is added.')
    parser.add_argument('-fv', '--false_vars', type = int, nargs = 1, metavar = 0, default = [0], help='The number of false variables (variables that are present but do not contribute to the y value).')
    myargs = parser.parse_args()
    create_random_data(myargs.data_size, myargs.mode[0], tuple(myargs.datarange), myargs.degrees, myargs.seed[0], myargs.coefficients, tuple(myargs.noise), myargs.false_vars[0])
