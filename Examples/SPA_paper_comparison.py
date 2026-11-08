import numpy as np
import matplotlib.pyplot as plt
from cv_final import CV_mse # TODO: this should be SPA.cv_final
import sys # To prevent SPA from printing info to the stdout
import os # To prevent SPA from printing info to the stdout; also for checkpointing

def SPA_paper_comparison(n_outer_loops = 2):
    # Setup
    l1_ratio = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99] # SPA's default list of values
    LCEN_cutoff = 4e-3
    var_list = np.arange(0, 251, 10) # What the SPA paper calls "noise level" is just the variance of the noise
    MSE_unbiased = np.empty((n_outer_loops, var_list.shape[0]))
    MSE_biased = np.empty_like(MSE_unbiased)
    MSE_cv = np.empty_like(MSE_unbiased)
    selected_degree = np.empty_like(MSE_unbiased, dtype = int)
    MSE_cv_limited = np.empty_like(MSE_unbiased)
    if os.path.exists('SPA_paper_comparison_MSE_cv_limited.csv'):
        temp = np.loadtxt('SPA_paper_comparison_MSE_unbiased.csv', comments = None, delimiter = ','); MSE_unbiased[:temp.shape[0], :] = temp.copy()
        temp = np.loadtxt('SPA_paper_comparison_MSE_biased.csv', comments = None, delimiter = ','); MSE_biased[:temp.shape[0], :] = temp.copy()
        temp = np.loadtxt('SPA_paper_comparison_MSE_cv.csv', comments = None, delimiter = ','); MSE_cv[:temp.shape[0], :] = temp.copy()
        temp = np.loadtxt('SPA_paper_comparison_selected_degree.csv', comments = None, delimiter = ',', dtype = int); selected_degree[:temp.shape[0], :] = temp.copy()
        temp = np.loadtxt('SPA_paper_comparison_MSE_cv_limited.csv', comments = None, delimiter = ','); MSE_cv_limited[:temp.shape[0], :] = temp.copy()
        outer_loops_to_skip = temp.shape[0]
    else:
        outer_loops_to_skip = 0
    # Running n_outer_loops simulations for MSE values
    for idx_outer in range(outer_loops_to_skip, n_outer_loops):
        print(f'Current outer loop: {idx_outer+1:3}/{n_outer_loops}' + ' '*20)
        rng = np.random.default_rng(idx_outer)
        X_train = rng.normal(0, np.sqrt(5), (30, 1))
        y_train = X_train + 0.5*X_train**2 + 0.1*X_train**3 + 0.05*X_train**4
        X_test = rng.normal(0, np.sqrt(5), (1000, 1))
        y_test = X_test + 0.5*X_test**2 + 0.1*X_test**3 + 0.05*X_test**4
        old_stdout = sys.stdout # To prevent SPA from printing info to the stdout; backup current stdout
        sys.stdout = open(os.devnull, "w")
        for idx_noise, var in enumerate(var_list):
            y_train_noisy = y_train + rng.normal(0, np.sqrt(var), (30, 1))
            # Unbiased model - only degree = 4
            _, _, _, _, MSE_unbiased[idx_outer, idx_noise], _, _, _ = CV_mse('LCEN', X_train, y_train_noisy, X_test, y_test,
                        cv_type = 'KFold', K_fold = 3, alpha = 20, l1_ratio = l1_ratio, degree = [4], trans_type = 'poly', LCEN_cutoff = LCEN_cutoff)
            MSE_unbiased[idx_outer, idx_noise] /= y_train_noisy.std()**2
            # Biased model - only degree = 2
            _, _, _, _, MSE_biased[idx_outer, idx_noise], _, _, _ = CV_mse('LCEN', X_train, y_train_noisy, X_test, y_test,
                        cv_type = 'KFold', K_fold = 3, alpha = 20, l1_ratio = l1_ratio, degree = [2], trans_type = 'poly', LCEN_cutoff = LCEN_cutoff)
            MSE_biased[idx_outer, idx_noise] /= y_train_noisy.std()**2
            # CV model - degrees from 1 to 10
            hyperparams, _, _, _, MSE_cv[idx_outer, idx_noise], _, _, _ = CV_mse('LCEN', X_train, y_train_noisy, X_test, y_test,
                        cv_type = 'KFold', K_fold = 3, alpha = 20, l1_ratio = l1_ratio, degree = list(range(1, 11)), trans_type = 'poly', LCEN_cutoff = LCEN_cutoff)
            MSE_cv[idx_outer, idx_noise] /= y_train_noisy.std()**2
            selected_degree[idx_outer, idx_noise] = hyperparams['degree']
            # CV limited order model - degree 2 or 4
            _, _, _, _, MSE_cv_limited[idx_outer, idx_noise], _, _, _ = CV_mse('LCEN', X_train, y_train_noisy, X_test, y_test,
                        cv_type = 'KFold', K_fold = 3, alpha = 20, l1_ratio = l1_ratio, degree = [2, 4], trans_type = 'poly', LCEN_cutoff = LCEN_cutoff)
            MSE_cv_limited[idx_outer, idx_noise] /= y_train_noisy.std()**2
        sys.stdout = old_stdout # reset old stdout so we can print info on the current outer loop
        if idx_outer%5 == 4: # Save every 5 outer loops # TODO: perhaps save as .npy files, since they may take less disk space
            np.savetxt('SPA_paper_comparison_MSE_unbiased.csv', MSE_unbiased[:idx_outer+1], delimiter = ',', comments = '')
            np.savetxt('SPA_paper_comparison_MSE_biased.csv', MSE_biased[:idx_outer+1], delimiter = ',', comments = '')
            np.savetxt('SPA_paper_comparison_MSE_cv.csv', MSE_cv[:idx_outer+1], delimiter = ',', comments = '')
            np.savetxt('SPA_paper_comparison_selected_degree.csv', selected_degree[:idx_outer+1], delimiter = ',', comments = '', fmt = '%d')
            np.savetxt('SPA_paper_comparison_MSE_cv_limited.csv', MSE_cv_limited[:idx_outer+1], delimiter = ',', comments = '')
    # Plotting
    plt.rcParams.update({'lines.markersize': 7, 'font.size': 30})
    fig, ax = plt.subplots(figsize = (12, 9), dpi = 300)
    ax.plot(var_list, np.median(MSE_unbiased, axis = 0), 'o-', label = 'Unbiased Model')
    ax.plot(var_list, np.median(MSE_biased, axis = 0), 's-', label = 'Biased Model')
    ax.plot(var_list, np.median(MSE_cv, axis = 0), '^-', label = 'CV Model')
    ax.plot(var_list, np.median(MSE_cv_limited, axis = 0), 'P-', label = 'CV Limited Model')
    ax.set_xlabel('Noise variance ' + r'$\sigma^2$')
    ax.set_xlim([-1, var_list[-1]+1])
    ax.set_ylabel('Median test MSE')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'SPA_paper_comparison_{n_outer_loops}runs.svg', bbox_inches = 'tight')
    # Saving a version that goes only to 200 for easier comparison with the original SPA paper figure
    ax.set_xlim([-1, 201])
    plt.savefig(f'SPA_paper_comparison_{n_outer_loops}runs_zoomed.svg', bbox_inches = 'tight')
    plt.close()
    # Saving the interquartile range
    plt.rcParams.update({'lines.markersize': 4, 'font.size': 20})
    fig, ax = plt.subplots(figsize = (12, 9), dpi = 300)
    output = ax.plot(var_list, np.percentile(MSE_unbiased, 25, axis = 0), 's-', label = 'Unbiased Model')
    ax.plot(var_list, np.percentile(MSE_unbiased, 75, axis = 0), '^-', label = '', color = output[0].get_color())
    output = ax.plot(var_list, np.percentile(MSE_biased, 25, axis = 0), 's-', label = 'Biased Model')
    ax.plot(var_list, np.percentile(MSE_biased, 75, axis = 0), '^-', label = '', color = output[0].get_color())
    output = ax.plot(var_list, np.percentile(MSE_cv, 25, axis = 0), 's-', label = 'CV Model')
    ax.plot(var_list, np.percentile(MSE_cv, 75, axis = 0), '^-', label = '', color = output[0].get_color())
    output = ax.plot(var_list, np.percentile(MSE_cv_limited, 25, axis = 0), 's-', label = 'CV Limited Model')
    ax.plot(var_list, np.percentile(MSE_cv_limited, 75, axis = 0), '^-', label = '', color = output[0].get_color())
    ax.set_xlabel('Noise variance ' + r'$\sigma^2$')
    ax.set_xlim([-1, 201])
    ax.set_ylabel('25% or 75% quartile Test MSE ')
    ax.legend(bbox_to_anchor = (0.99, 0.30))#title = r'25% = $\blacksquare$, 75% = $\blacktriangle$')
    plt.tight_layout()
    plt.savefig(f'SPA_paper_comparison_{n_outer_loops}runs_zoomed_interquartile.svg', bbox_inches = 'tight')
    # Across the n_outer_loops, how often was a given degree picked when the noise was fixed? -- degree heatmap
    plt.rcParams.update({'font.size': 30})
    selected_degree_heatmap = np.empty((var_list.shape[0], 10))
    for idx_noise in range(var_list.shape[0]):
        selected_degree_heatmap[idx_noise, :] = (selected_degree[:, idx_noise].reshape(-1, 1) == np.arange(1,11)).sum(axis=0)
    plt.rcParams.update({'font.size': 10})
    fig, ax = plt.subplots(dpi = 300)
    im = ax.imshow(selected_degree_heatmap)
    ax.set_xticks(np.arange(1, 11)-1, labels = np.arange(1, 11))
    ax.set_xlabel('Selected Degree')
    ax.set_yticks(np.arange(var_list.shape[0]), labels = var_list)
    ax.set_ylabel('Noise variance ' + r'$\sigma^2$')
    cbar = ax.figure.colorbar(im)
    plt.tight_layout()
    plt.savefig(f'SPA_paper_comparison_degree-heatmap_{n_outer_loops}runs.svg', bbox_inches = 'tight')

if __name__ == '__main__':
    # Input setup
    import argparse
    parser = argparse.ArgumentParser(description = 'Repeats the test done in the SPA paper to compare the median MSE of 4 types of CV at varying noises.')
    parser.add_argument('-n', '--n_outer_loops', type = int, nargs = 1, metavar = 2, default = [2], help = 'The number of outer loops to run (so that the final result is the median). The SPA paper used 3,000 and we use 10,000.')
    myargs = parser.parse_args()
    SPA_paper_comparison(myargs.n_outer_loops[0])
