"""
n = int(sys.argv[1])
time_limit = int(sys.argv[2])
seed = int(sys.argv[3])
method = str(sys.argv[4])

run for
n: 10, 20, 30, 40, 50
time_limit: 60, 300, 600
seed: 0, 1, ..., 49
method: 'classification', 'regression'
"""

import warnings
import shutil
import sys
import os

import numpy as np
from tqdm import tqdm
from autogluon.tabular import TabularPredictor, TabularDataset
import pandas as pd

n = int(sys.argv[1])
time_limit = int(sys.argv[2])
seed = int(sys.argv[3])
method = str(sys.argv[4])

# Define results path and create directory.
path = './results_blobs_autogluon/'
path += method + '/'
path += str(n) + '/'
path += str(time_limit) + '/'
path += str(seed) + '/'
if not os.path.exists(path):
    os.makedirs(path)


def snr_score(estimator, x_test, y_test, permutations=None):
    # pred = estimator.predict(x_test).reshape(-1, )
    pred = estimator.predict_proba(x_test)

    pred = np.array(pred)
    y_test = np.array(y_test).reshape(-1, )

    p_samp = pred[y_test ==1]
    q_samp = pred[y_test == 0]
    # print(len(p_samp), len(q_samp))
    c = len(p_samp) / (len(p_samp) + len(q_samp))
    # c = 1-c
    signal = (np.mean(p_samp) - np.mean(q_samp))
    if permutations is None:
        if c == 1 or c == 0:
            return -500
        noise = np.sqrt(1 / c * np.var(p_samp) + 1 / (1 - c) * np.var(q_samp))
        if noise == 0:
            return - 500
        snr = signal / noise
        # check for nan
        if snr != snr:
            return -500
        else:
            return snr
    else:
        p = 0
        for i in range(permutations):
            np.random.shuffle(pred)
            p_samp = pred[y_test == 1]
            q_samp = pred[y_test == 0]
            signal_perm = np.mean(p_samp) - np.mean(q_samp)

            if signal <= float(signal_perm):
                p += float(1 / permutations)
        # print(signal, p)
        return p  # this is the corresponding SNR


def sample_blobs(n, rows=3, cols=3, sep=1, rs=np.random):
    """Generate Blob-S for testing type-I error."""
    correlation = 0
    # generate within-blob variation
    mu = np.zeros(2)
    sigma = np.eye(2)
    X = rs.multivariate_normal(mu, sigma, size=n)
    corr_sigma = np.array([[1, correlation], [correlation, 1]])
    Y = rs.multivariate_normal(mu, corr_sigma, size=n)
    # assign to blobs
    X[:, 0] += rs.randint(rows, size=n) * sep
    X[:, 1] += rs.randint(cols, size=n) * sep
    Y[:, 0] += rs.randint(rows, size=n) * sep
    Y[:, 1] += rs.randint(cols, size=n) * sep
    return X, Y


def sample_blobs_Q(N1, sigma_mx_2, rows=3, cols=3, rs=None):
    """Generate Blob-D for testing type-II error (or test power)."""
    mu = np.zeros(2)
    sigma = np.eye(2) * 0.03
    # rs = np.random
    X = rs.multivariate_normal(mu, sigma, size=N1)
    Y = rs.multivariate_normal(mu, np.eye(2), size=N1)
    # assign to blobs
    X[:, 0] += rs.randint(rows, size=N1)
    X[:, 1] += rs.randint(cols, size=N1)
    Y_row = rs.randint(rows, size=N1)
    Y_col = rs.randint(cols, size=N1)
    locs = [[0,0],[0,1],[0,2],[1,0],[1,1],[1,2],[2,0],[2,1],[2,2]]
    for i in range(9):
        corr_sigma = sigma_mx_2[i]
        L = np.linalg.cholesky(corr_sigma)
        ind = np.expand_dims((Y_row == locs[i][0]) & (Y_col == locs[i][1]), 1)
        ind2 = np.concatenate((ind, ind), 1)
        Y = np.where(ind2, np.matmul(Y,L) + locs[i], Y)
    return X, Y



# blobs definitions used by Liu et al. 2020
sigma_mx_2_standard = np.array([[0.03, 0], [0, 0.03]])
sigma_mx_2 = np.zeros([9,2,2])
for i in range(9):
    sigma_mx_2[i] = sigma_mx_2_standard
    if i < 4:
        sigma_mx_2[i][0 ,1] = -0.02 - 0.002 * i
        sigma_mx_2[i][1, 0] = -0.02 - 0.002 * i
    if i==4:
        sigma_mx_2[i][0, 1] = 0.00
        sigma_mx_2[i][1, 0] = 0.00
    if i>4:
        sigma_mx_2[i][1, 0] = 0.02 + 0.002 * (i-5)
        sigma_mx_2[i][0, 1] = 0.02 + 0.002 * (i-5)


# n_list = [10, 20, 30, 40, 50] # number of samples in per mode

np.random.seed(1)
rng = np.random.RandomState(seed=seed) # used to draw the samples
n_per_class = 9*n
results_witness = []
warnings.filterwarnings("ignore")
pbar = tqdm(range(10))
for i in pbar:
    s1,s2 = sample_blobs_Q(n_per_class, sigma_mx_2, rs=rng)
    # # print(s1[:5])
    s1test, s2test = sample_blobs_Q(n_per_class, sigma_mx_2, rs=rng)

    # # this would be to investigate type-I error

    # s1,s2 = sample_blobs(n_per_class, rs=rng)
    # s1test, s2test = sample_blobs(n_per_class, rs=rng)

    X_train = np.concatenate((s1, s2))
    X_test = np.concatenate((s1test, s2test))
    Y_train = np.concatenate(([1 ] * n_per_class, [0 ] * n_per_class))
    Y_test = np.concatenate(([1 ] * n_per_class, [0 ] * n_per_class))
    # this is needed, since the cross-validation does not shuffle before creating the batches
    shuffle_train = np.random.permutation(2*n_per_class)
    shuffle_test = np.random.permutation(2*n_per_class)
    X_train, Y_train = X_train[shuffle_train], Y_train[shuffle_train]
    X_test, Y_test = X_test[shuffle_test], Y_test[shuffle_test]
    # print(Y_train.shape, X_train.shape)
    df_train = pd.DataFrame({"data0": X_train[:, 0], "data1": X_train[:, 1], "label": Y_train})
    df_test = pd.DataFrame({"data0": X_test[:, 0], "data1": X_test[:, 1], "label": Y_test})

    train_data = TabularDataset(df_train)
    test_data = TabularDataset(df_test)
    # print(train_data[:5])
    # predictor = TabularPredictor(label="label", problem_type="regression", eval_metric="mean_squared_error",
    #                              verbosity=0).fit(train_data, presets='best_quality', time_limit=60)
    if method == 'classification':
        predictor = TabularPredictor(label="label", problem_type="binary", eval_metric="accuracy",
                                     verbosity=0).fit(train_data, presets='best_quality', time_limit=time_limit)
    elif method == 'regression':
        predictor = TabularPredictor(label="label", problem_type="regression", eval_metric="mean_squared_error",
                                     verbosity=0).fit(train_data, presets='best_quality', time_limit=time_limit)
    else:
        print("No valid method selected. Should be <classification> or <regression>")


#
    # snr = snr_score(grid_search.best_estimator_, X_test, Y_test)
    # tau = np.sqrt(len(X_test)) * snr
    # p = 1 - norm.cdf(tau)
    # with permutations we return directly the pvalue
    p = snr_score(predictor, test_data, Y_test, permutations=300)
    model_path = predictor.path
    shutil.rmtree(model_path[:-1])
    results_witness.append(1) if p < 0.05 else results_witness.append(0)

    pbar.set_description("n= %.0f" % n_per_class + " witness power: %.4f" % np.mean(results_witness) + " current p-value %.4f" %p)

with open(f'{path}results.npy', 'wb') as f:
    np.save(f, results_witness)
