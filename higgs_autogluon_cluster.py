"""
Requires the HIGGS dataset to be stored in the same directory
Download pickled file from here:
https://drive.google.com/open?id=1sHIIFCoHbauk6Mkb6e8a_tp1qnvuUOCc

script takes 3 arguments: n (samplesize), time_limit, and seed.
Script runs 10 repetitions.
It should be run for all combinations of:
n: 5000, 4000, 2500, 1500, 1000, 500
time_limit: 300, 120, 20
seed: 0, 1, ..., 49
"""

import warnings

import numpy as np
import pickle
from tqdm import tqdm

import sys
import os
import shutil

from autogluon.tabular import TabularPredictor, TabularDataset

import pandas as pd

n = int(sys.argv[1])
if sys.argv[2] == 'None':
    time_limit = None
else:
    time_limit = int(sys.argv[2])
seed = int(sys.argv[3])
if len(sys.argv) > 4:
    control = sys.argv[4] in ["control"]
else:
    control = False

# Define results path and create directory.
path = sys.argv[4]
path += str(n) + '/'
path += str(time_limit) + '/'
path += str(seed) + '/'
if not os.path.exists(path):
    os.makedirs(path)


print("train/test samples per dist %.0f" % n)
print("time limit in seconds %s" % str(time_limit))
print("seed %.f" % seed)


def snr_score(estimator, x_test, y_test, permutations=None):
    # pred = estimator.predict(x_test).reshape(-1, )
    pred = estimator.predict(x_test)

    pred = np.array(pred)
    y_test = np.array(y_test).reshape(-1, )

    p_samp = pred[y_test > 0]
    q_samp = pred[y_test < 0]
    # print(len(p_samp), len(q_samp))
    c = len(p_samp) / (len(p_samp) + len(q_samp))
    # c = 1-c
    signal = (np.mean(p_samp) - np.mean(q_samp))
    if permutations is None:
        if c == 1 or c==0:
            return -500
        noise = np.sqrt(1/c * np.var(p_samp) + 1/(1-c) * np.var(q_samp))
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
            p_samp = pred[y_test > 0]
            q_samp = pred[y_test < 0]
            signal_perm = np.mean(p_samp) - np.mean(q_samp)

            if signal <= float(signal_perm):
                p += float(1 / permutations)
        # print(signal, p)
        return p # this is the corresponding SNR



# Load data
data = pickle.load(open('./datasets/HIGGS_TST.pckl', 'rb'))
dataX = data[0]
dataY = data[1]
if control:
    # run type-I error control experiment as suggested by Liu et al
    print("Experiment for Type-I error control")
    dataX = data[0]
    dataY = data[0]

del data

np.random.seed(seed)
rng = np.random.RandomState(seed=seed)
results_witness = []
# warnings.filterwarnings("ignore")
pbar = tqdm(range(100))
for i in pbar:
    ## ---- Draw Data ---- ###
    # Generate Higgs (P,Q)
    N1_T = dataX.shape[0]
    N2_T = dataY.shape[0]
    ind1 = rng.choice(N1_T, n, replace=False)
    ind2 = rng.choice(N2_T, n, replace=False)
    s1 = dataX[ind1,:4]
    s2 = dataY[ind2,:4]
    ind1 = rng.choice(N1_T, n, replace=False)
    ind2 = rng.choice(N2_T, n, replace=False)
    s1test = dataX[ind1,:4]
    s2test = dataY[ind2,:4]

    X_train = np.concatenate((s1, s2))
    X_test = np.concatenate((s1test, s2test))
    Y_train = np.concatenate(([1] * n, [-1] * n))
    Y_test = np.concatenate(([1] * n, [-1] * n))
    shuffle_train = rng.permutation(2 * n)
    shuffle_test = rng.permutation(2 * n)
    X_train, Y_train = X_train[shuffle_train], Y_train[shuffle_train]
    X_test, Y_test = X_test[shuffle_test], Y_test[shuffle_test]

    df_train = pd.DataFrame({"data0": X_train[:, 0], "data1": X_train[:, 1], "data2": X_train[:, 2], "data3": X_train[:, 3], "label": Y_train})
    df_test = pd.DataFrame({"data0": X_test[:, 0], "data1": X_test[:, 1], "data2": X_test[:, 2], "data3": X_test[:, 3], "label": Y_test})

    train_data = TabularDataset(df_train)
    test_data = TabularDataset(df_test)
    predictor = TabularPredictor(label="label", problem_type="regression", eval_metric="mean_squared_error",
                                 verbosity=0).fit(train_data, presets='best_quality', time_limit=time_limit)

    #
    # snr = snr_score(grid_search.best_estimator_, X_test, Y_test)
    # tau = np.sqrt(len(X_test)) * snr
    # p = 1 - norm.cdf(tau)
    # with permutations we return directly the pvalue
    p = snr_score(predictor, test_data, Y_test, permutations=300)
    results_witness.append(1) if p < 0.05 else results_witness.append(0)
    model_path = predictor.path
    shutil.rmtree(model_path)

    pbar.set_description(
        "n= %.0f" % n + " witness power: %.4f" % np.mean(results_witness) + " current p-value %.4f" % p)
# save results
with open(f'{path}results.npy', 'wb') as f:
    np.save(f, results_witness)

