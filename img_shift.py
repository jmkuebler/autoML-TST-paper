"""
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--shift_type', type=str, required=True) # need to change this everywhere
parser.add_argument('--test_type', type=str, default='univ')
parser.add_argument('--time_limit', type=int, default=300, required=False)
parser.add_argument('--path', type= str, default='./results')
parser.add_argument('--pretrained', action='store_true')
parser.add_argument('--method', type=str, default='regression')
parser.add_argument('--discrete', action='store_true')
parser.add_argument('--mmd', action='store_true')
parser.add_argument('--mmdagg', action='store_true')

dataset: mnist, cifar10
shift_type: orig, small_gn_shift, medium_gn_shift, large_gn_shift,
    adversarial_shift, small_image_shift, medium_image_shift,
    large_image_shift, medium_img_shift+ko_shift,
    only_zero_shift+medium_img_shift
test_type: univ
Runtime limit time_limit: 60, 300, 600
Folder to store results path: ./results
Flag whether to use feature from pretrained model pretrained: True, False
method: binary, regression
Flag whether to use discrete features discrete: True, False
Flag whether to use deep-mmd instead of witness: True, False
Flag whether to use mmdagg instead of witness: True, False
"""


# -------------------------------------------------
# IMPORTS
# -------------------------------------------------

import numpy as np
from tensorflow import set_random_seed
seed = 1
np.random.seed(seed)
set_random_seed(seed)

import tempfile
import keras.models

from shift_detector import *
from shift_locator import *
from shift_applicator import *
from data_utils import *
from shared_utils import *
from deep_mmd import deep_mmd
import os
import argparse

import matplotlib.pyplot as plt
from matplotlib import rc

from mmd_agg.tests import mmdagg as mmdagg_test

# -------------------------------------------------
# PLOTTING HELPERS
# -------------------------------------------------


rc('font',**{'family':'serif','serif':['Times']})
#rc('text', usetex=True)
rc('axes', labelsize=22)
rc('xtick', labelsize=22)
rc('ytick', labelsize=22)
rc('legend', fontsize=13)

#mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']


def clamp(val, minimum=0, maximum=255):
    if val < minimum:
        return minimum
    if val > maximum:
        return maximum
    return val


def colorscale(hexstr, scalefactor):
    hexstr = hexstr.strip('#')

    if scalefactor < 0 or len(hexstr) != 6:
        return hexstr

    r, g, b = int(hexstr[:2], 16), int(hexstr[2:4], 16), int(hexstr[4:], 16)

    r = clamp(r * scalefactor)
    g = clamp(g * scalefactor)
    b = clamp(b * scalefactor)

    return "#%02x%02x%02x" % (int(r), int(g), int(b))


def errorfill(x, y, yerr, color=None, alpha_fill=0.2, ax=None, fmt='-o', label=None):
    ax = ax if ax is not None else plt.gca()
    if color is None:
        color = next(ax._get_lines.prop_cycler)['color']
    if np.isscalar(yerr) or len(yerr) == len(y):
        ymin = y - yerr
        ymax = y + yerr
    elif len(yerr) == 2:
        ymin, ymax = yerr
    ax.semilogx(x, y, fmt, color=color, label=label)
    ax.fill_between(x, np.clip(ymax, 0, 1), np.clip(ymin, 0, 1), color=color, alpha=alpha_fill)


def make_keras_picklable():
    def __getstate__(self):
        model_str = ""
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            keras.models.save_model(self, fd.name, overwrite=True)
            model_str = fd.read()
        d = { 'model_str': model_str }
        return d

    def __setstate__(self, state):
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            fd.write(state['model_str'])
            fd.flush()
            model = keras.models.load_model(fd.name)
        self.__dict__ = model.__dict__


    cls = keras.models.Model
    cls.__getstate__ = __getstate__
    cls.__setstate__ = __setstate__


linestyles = ['-', '-.', '--', ':']
brightness = [1.25, 1.0, 0.75, 0.5]
format = ['-o', '-h', '-p', '-s', '-D', '-<', '->', '-X']
markers = ['o', 'h', 'p', 's', 'D', '<', '>', 'X']
colors_old = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
              '#bcbd22', '#17becf']
colors = ['#2196f3', '#f44336', '#9c27b0', '#64dd17', '#009688', '#ff9800', '#795548', '#607d8b']

# -------------------------------------------------
# CONFIG
# -------------------------------------------------

make_keras_picklable()

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--shift_type', type=str, required=True) # need to change this everywhere
parser.add_argument('--test_type', type=str, default='univ')
parser.add_argument('--time_limit', type=int, default=300, required=False)
parser.add_argument('--path', type= str, default='./results')
parser.add_argument('--pretrained', action='store_true')
parser.add_argument('--method', type=str, default='regression')
parser.add_argument('--discrete', action='store_true')
parser.add_argument('--mmd', action='store_true')
parser.add_argument('--mmdagg', action='store_true')


args = parser.parse_args()

dataset = args.dataset
shift_type = args.shift_type
test_type = args.test_type
time_limit = int(args.time_limit)
pretrained = args.pretrained
print('Using pretrained model: %s' % pretrained)
method = args.method
if method == 'binary':
    discrete = args.discrete
    use_prob = not discrete
else:
    use_prob = False
# if len(sys.argv) > 7:
#     method = sys.argv[7]
#     discrete = sys.argv[8] == 'True'
#     use_prob = not discrete
# else:
#     method = 'regression'
#     use_prob = False
mmd = args.mmd
mmdagg = args.mmdagg

print('Fitting method to use: %s' % method)
print('Use probability: %s' % str(use_prob))

# Define results path and create directory.
path = args.path
if mmdagg:
    path += '_mmdagg'
path += '/' + test_type + '/'
path += dataset + '_'
path += shift_type + '/'
if not os.path.exists(path):
    os.makedirs(path)

# Define DR methods.
dr_techniques = []
# dr_techniques = [DimensionalityReduction.NoRed.value, DimensionalityReduction.PCA.value, DimensionalityReduction.SRP.value, DimensionalityReduction.UAE.value, DimensionalityReduction.TAE.value, DimensionalityReduction.BBSDs.value, DimensionalityReduction.BBSDh.value]
dr_techniques = []
if test_type == 'multiv':
    dr_techniques = [DimensionalityReduction.NoRed.value, DimensionalityReduction.PCA.value, DimensionalityReduction.SRP.value, DimensionalityReduction.UAE.value, DimensionalityReduction.TAE.value, DimensionalityReduction.BBSDs.value]
if test_type == 'univ':
    dr_techniques_plot = dr_techniques.copy()
    dr_techniques_plot.append(DimensionalityReduction.Classif.value)
else:
    dr_techniques_plot = dr_techniques.copy()

# Define test types and general test sample sizes.
test_types = [td.value for td in TestDimensionality]
if test_type == 'multiv':
    od_tests = []
    md_tests = [MultidimensionalTest.MMD.value]
    samples = [10, 20, 50, 100, 200, 500, 1000]
else:
    od_tests = [OnedimensionalTest.KS.value]
    md_tests = []
    samples = [10, 20, 50, 100, 200, 500, 1000, 10000]
    if mmdagg:
        samples = [10, 20, 50, 100, 200]
difference_samples = 20

# Number of random runs to average results over.
random_runs = 5


# Significance level.
sign_level = 0.05

# Whether to calculate accuracy for malignancy quantification.
# calc_acc = True
calc_acc = False

# Define shift types.
if shift_type == 'small_gn_shift':
    shifts = ['small_gn_shift_0.1',
              'small_gn_shift_0.5',
              'small_gn_shift_1.0']
elif shift_type == 'medium_gn_shift':
    shifts = ['medium_gn_shift_0.1',
              'medium_gn_shift_0.5',
              'medium_gn_shift_1.0']
elif shift_type == 'large_gn_shift':
    shifts = ['large_gn_shift_0.1',
              'large_gn_shift_0.5',
              'large_gn_shift_1.0']
elif shift_type == 'adversarial_shift':
    shifts = ['adversarial_shift_0.1',
              'adversarial_shift_0.5',
              'adversarial_shift_1.0']
elif shift_type == 'ko_shift':
    shifts = ['ko_shift_0.1',
              'ko_shift_0.5',
              'ko_shift_1.0']
    if test_type == 'univ' and not mmdagg:
        samples = [10, 20, 50, 100, 200, 500, 1000, 9000]
elif shift_type == 'orig':
    shifts = ['rand', 'orig']
    brightness = [1.25, 0.75]
elif shift_type == 'small_image_shift':
    shifts = ['small_img_shift_0.1',
              'small_img_shift_0.5',
              'small_img_shift_1.0']
elif shift_type == 'medium_image_shift':
    shifts = ['medium_img_shift_0.1',
              'medium_img_shift_0.5',
              'medium_img_shift_1.0']
elif shift_type == 'large_image_shift':
    shifts = ['large_img_shift_0.1',
              'large_img_shift_0.5',
              'large_img_shift_1.0']
elif shift_type == 'medium_img_shift+ko_shift':
    shifts = ['medium_img_shift_0.5+ko_shift_0.1',
              'medium_img_shift_0.5+ko_shift_0.5',
              'medium_img_shift_0.5+ko_shift_1.0']
    if test_type == 'univ' and not mmdagg:
        samples = [10, 20, 50, 100, 200, 500, 1000, 9000]
elif shift_type == 'only_zero_shift+medium_img_shift':
    shifts = ['only_zero_shift+medium_img_shift_0.1',
              'only_zero_shift+medium_img_shift_0.5',
              'only_zero_shift+medium_img_shift_1.0']
    if not mmdagg:
        samples = [10, 20, 50, 100, 200, 500, 1000]
else:
    shifts = []
    
if dataset == 'coil100' and test_type == 'univ':
    samples = [10, 20, 50, 100, 200, 500, 1000, 2400]

if dataset == 'mnist_usps':
    samples = [10, 20, 50, 100, 200, 500, 1000]

# -------------------------------------------------
# PIPELINE START
# -------------------------------------------------

# Stores p-values for all experiments of a shift class.
samples_shifts_rands_dr_tech = np.ones((len(samples), len(shifts), random_runs, len(dr_techniques_plot))) * (-1)

red_dim = -1
red_models = [None] * len(DimensionalityReduction)

# Iterate over all shifts in a shift class.
for shift_idx, shift in enumerate(shifts):
    print(shift)

    shift_path = path + shift + '/'
    if not os.path.exists(shift_path):
        os.makedirs(shift_path)

    # Stores p-values for a single shift.
    rand_run_p_vals = np.ones((len(samples), len(dr_techniques_plot), random_runs)) * (-1)

    # Stores accuracy values for malignancy detection.
    val_accs = np.ones((random_runs, len(samples))) * (-1)
    te_accs = np.ones((random_runs, len(samples))) * (-1)
    dcl_accs = np.ones((len(samples), random_runs)) * (-1)

    # Average over a few random runs to quantify robustness.
    for rand_run in range(0, random_runs-1):
        rand_run = int(rand_run)
        print("Random run %s" % rand_run)

        rand_run_path = shift_path + str(rand_run) + '/'
        if not os.path.exists(rand_run_path):
            os.makedirs(rand_run_path)

        np.random.seed(rand_run)
        set_random_seed(rand_run)

        # Load data.
        (X_tr_orig, y_tr_orig), (X_val_orig, y_val_orig), (X_te_orig, y_te_orig), orig_dims, nb_classes = \
            import_dataset(dataset, shuffle=True)
        X_tr_orig = normalize_datapoints(X_tr_orig, 255.)
        X_te_orig = normalize_datapoints(X_te_orig, 255.)
        X_val_orig = normalize_datapoints(X_val_orig, 255.)

        # Apply shift.
        if shift == 'orig':
            print('Original')
            (X_tr_orig, y_tr_orig), (X_val_orig, y_val_orig), (X_te_orig, y_te_orig), orig_dims, nb_classes = import_dataset(dataset)
            X_tr_orig = normalize_datapoints(X_tr_orig, 255.)
            X_te_orig = normalize_datapoints(X_te_orig, 255.)
            X_val_orig = normalize_datapoints(X_val_orig, 255.)
            X_te_1 = X_te_orig.copy()
            y_te_1 = y_te_orig.copy()
        else:
            (X_te_1, y_te_1) = apply_shift(X_te_orig, y_te_orig, shift, orig_dims, dataset)

        X_te_2 , y_te_2 = random_shuffle(X_te_1, y_te_1)

        # Check detection performance for different numbers of samples from test.
        for si, sample in enumerate(samples):

            print("Sample %s" % sample)

            sample_path = rand_run_path + str(sample) + '/'
            if not os.path.exists(sample_path):
                os.makedirs(sample_path)

            X_te_3 = X_te_2[:sample,:]
            y_te_3 = y_te_2[:sample]

            if test_type == 'multiv':
                X_val_3 = X_val_orig[:1000,:]
                y_val_3 = y_val_orig[:1000]
            else:
                X_val_3 = X_val_orig[:sample,:]
                y_val_3 = y_val_orig[:sample]

            X_tr_3 = np.copy(X_tr_orig)
            y_tr_3 = np.copy(y_tr_orig)

            # Detect shift.
            shift_detector = ShiftDetector(dr_techniques, test_types, od_tests, md_tests, sign_level, red_models,
                                           sample, dataset)
            (od_decs, ind_od_decs, ind_od_p_vals), \
            (md_decs, ind_md_decs, ind_md_p_vals), \
            red_dim, red_models, val_acc, te_acc = shift_detector.detect_data_shift(X_tr_3, y_tr_3, X_val_3, y_val_3,
                                                                                    X_te_3, y_te_3, orig_dims,
                                                                                    nb_classes)
            
            val_accs[rand_run, si] = val_acc
            te_accs[rand_run, si] = te_acc

            if test_type == 'multiv':
                print("Shift decision: ", ind_md_decs.flatten())
                print("Shift p-vals: ", ind_md_p_vals.flatten())

                rand_run_p_vals[si,:,rand_run] = ind_md_p_vals.flatten()
            else:
                print("Shift decision: ", ind_od_decs.flatten())
                print("Shift p-vals: ", ind_od_p_vals.flatten())
                
                if DimensionalityReduction.Classif.value not in dr_techniques_plot:
                    rand_run_p_vals[si,:,rand_run] = ind_od_p_vals.flatten()
                    continue
                if not (mmd or mmdagg):
                    # Characterize shift via domain classifier.
                    if pretrained: # work with the output of a pretrained network as is done for BBSDs
                        print("Uses pretrained model")
                        # load pretrained model
                        shift_reductor = ShiftReductor(X_tr_3, y_tr_3, None, None, DimensionalityReduction(DimensionalityReduction.BBSDs.value), orig_dims, dataset, dr_amount=32)
                        pretrained_model = shift_reductor.fit_reductor()
                        X_tr_red = shift_reductor.reduce(pretrained_model, X_val_3)
                        X_te_red = shift_reductor.reduce(pretrained_model, X_te_3)
                        shift_locator = ShiftLocator(orig_dims, dc=DifferenceClassifier.AUTOGLUON, sign_level=sign_level)
                        model, score, (X_tr_dcl, y_tr_dcl, y_tr_old, X_te_dcl, y_te_dcl, y_te_old) = shift_locator.build_model(X_tr_red, y_val_3, X_te_red, y_te_3, time_limit=time_limit, method=method)
                        test_indices, test_perc, dec, p_val = shift_locator.most_likely_shifted_samples(model, X_te_dcl, y_te_dcl, use_prob=use_prob)
                    else:
                        shift_locator = ShiftLocator(orig_dims, dc=DifferenceClassifier.AUTOGLUON, sign_level=sign_level)
                        print(len(X_te_3))
                        model, score, (X_tr_dcl, y_tr_dcl, y_tr_old, X_te_dcl, y_te_dcl, y_te_old) = shift_locator.build_model(X_tr_3, y_tr_3, X_te_3, y_te_3, time_limit=time_limit, method=method)
                        test_indices, test_perc, dec, p_val = shift_locator.most_likely_shifted_samples(model, X_te_dcl, y_te_dcl, use_prob=use_prob)
                elif mmd:
                    # run deep MMD test by Liu et al
                    # make sample for MMD test.
                    sample_p = X_tr_3[:len(X_te_3)]
                    sample_q = X_te_3
                    dec, p_val = deep_mmd(sample_p, sample_q, sign_level=sign_level, datset=dataset)

                elif mmdagg:
                    sample_p = X_tr_3[:len(X_te_3)]
                    sample_q = X_te_3
                    # # check type I:
                    # joint = np.concatenate((sample_p, sample_q))
                    # np.random.shuffle(joint)
                    # sample_p, sample_q = joint[:len(joint)//2], joint[len(joint)//2:]

                    dec = mmdagg_test(seed=0, X=sample_p, Y=sample_q, alpha=sign_level, kernel_type="gaussian",
                                      approx_type="permutation", weights_type="uniform", l_minus=10, l_plus=20, B1=500,
                                      B2=500, B3=100)
                    p_val = dec  # because the method does not compute p-values we just hack it like this.

                rand_run_p_vals[si,:,rand_run] = np.append(ind_od_p_vals.flatten(), p_val)

        for dr_idx, dr in enumerate(dr_techniques_plot):
            plt.semilogx(np.array(samples), rand_run_p_vals[:,dr_idx,rand_run], format[dr], color=colors[dr], label="%s" % DimensionalityReduction(dr).name)
        plt.axhline(y=sign_level, color='k')
        plt.xlabel('Number of samples from test')
        plt.ylabel('p-value')
        plt.savefig("%s/dr_sample_comp_noleg.pdf" % rand_run_path, bbox_inches='tight')
        plt.legend()
        plt.savefig("%s/dr_sample_comp.pdf" % rand_run_path, bbox_inches='tight')
        plt.clf()

        np.savetxt("%s/dr_method_p_vals.csv" % rand_run_path, rand_run_p_vals[:,:,rand_run], delimiter=",")

    mean_p_vals = np.mean(rand_run_p_vals, axis=2)
    std_p_vals = np.std(rand_run_p_vals, axis=2)

    mean_val_accs = np.mean(val_accs, axis=0)
    std_val_accs = np.std(val_accs, axis=0)

    mean_te_accs = np.mean(te_accs, axis=0)
    std_te_accs = np.std(te_accs, axis=0)

    if calc_acc and test_type == 'univ':
        mean_dcl_accs = []
        std_dcl_accs = []
        for si, sample in enumerate(samples):
            avg_val = 0
            elem_count = 0
            elem_list = []
            for rand_run in range(random_runs):
                current_val = dcl_accs[si, rand_run]
                if current_val == -1:
                    continue
                elem_list.append(current_val)
                avg_val = avg_val + current_val
                elem_count = elem_count + 1
            std_dcl_accs.append(np.std(np.array(elem_list)))
            if elem_count > 1:
                avg_val = avg_val / elem_count
            else:
                avg_val = -1
            mean_dcl_accs.append(avg_val)

        mean_dcl_accs = np.array(mean_dcl_accs)
        std_dcl_accs = np.array(std_dcl_accs)
        smpl_array = np.array(samples)
        min_one_indices = np.where(mean_dcl_accs == -1)

        print("mean_dcl_accs: ", mean_dcl_accs)
        print("std_dcl_accs: ", std_dcl_accs)
        print("smpl_array: ", smpl_array)

        print("-----------------")

        smpl_array = np.delete(smpl_array, min_one_indices)
        mean_dcl_accs = np.delete(mean_dcl_accs, min_one_indices)
        std_dcl_accs = np.delete(std_dcl_accs, min_one_indices)

        print("mean_dcl_accs: ", mean_dcl_accs)
        print("std_dcl_accs: ", std_dcl_accs)
        print("smpl_array: ", smpl_array)

        accs = np.ones((4, len(samples))) * (-1)
        accs[0] = mean_val_accs
        accs[1] = std_val_accs
        accs[2] = mean_te_accs
        accs[3] = std_te_accs

        dcl_accs = np.ones((3, len(smpl_array))) * (-1)
        dcl_accs[0] = smpl_array
        dcl_accs[1] = mean_dcl_accs
        dcl_accs[2] = std_dcl_accs

        np.savetxt("%s/accs.csv" % shift_path, accs, delimiter=",")
        np.savetxt("%s/dcl_accs.csv" % shift_path, dcl_accs, delimiter=",")

        errorfill(np.array(samples), mean_val_accs, std_val_accs, fmt='-o', color=colors[0], label=r"p")
        errorfill(np.array(samples), mean_te_accs, std_te_accs, fmt='-s', color=colors[3], label=r"q")
        if len(smpl_array) > 0:
            errorfill(smpl_array, mean_dcl_accs, std_dcl_accs, fmt='--X', color=colors[7], label=r"Classif")
        plt.xlabel('Number of samples from test')
        plt.ylabel('Accuracy')
        plt.savefig("%s/accs.pdf" % shift_path, bbox_inches='tight')
        plt.legend()
        plt.savefig("%s/accs_leg.pdf" % shift_path, bbox_inches='tight')
        plt.clf()
    

    for dr_idx, dr in enumerate(dr_techniques_plot):
        errorfill(np.array(samples), mean_p_vals[:,dr_idx], std_p_vals[:,dr_idx], fmt=format[dr], color=colors[dr], label="%s" % DimensionalityReduction(dr).name)
    plt.axhline(y=sign_level, color='k')
    plt.xlabel('Number of samples from test')
    plt.ylabel('p-value')
    plt.savefig("%s/dr_sample_comp_noleg.pdf" % shift_path, bbox_inches='tight')
    plt.legend()
    plt.savefig("%s/dr_sample_comp.pdf" % shift_path, bbox_inches='tight')
    plt.clf()

    for dr_idx, dr in enumerate(dr_techniques_plot):
        errorfill(np.array(samples), mean_p_vals[:,dr_idx], std_p_vals[:,dr_idx], fmt=format[dr], color=colors[dr])
        plt.xlabel('Number of samples from test')
        plt.ylabel('p-value')
        plt.axhline(y=sign_level, color='k', label='sign_level')
        plt.savefig("%s/%s_conf.pdf" % (shift_path, DimensionalityReduction(dr).name), bbox_inches='tight')
        plt.clf()

    np.savetxt("%s/mean_p_vals.csv" % shift_path, mean_p_vals, delimiter=",")
    np.savetxt("%s/std_p_vals.csv" % shift_path, std_p_vals, delimiter=",")

    for dr_idx, dr in enumerate(dr_techniques_plot):
        samples_shifts_rands_dr_tech[:,shift_idx,:,dr_idx] = rand_run_p_vals[:,dr_idx,:]

    np.save("%s/samples_shifts_rands_dr_tech.npy" % (path), samples_shifts_rands_dr_tech)

for dr_idx, dr in enumerate(dr_techniques_plot):
    dr_method_results = samples_shifts_rands_dr_tech[:,:,:,dr_idx]

    mean_p_vals = np.mean(dr_method_results, axis=2)
    std_p_vals = np.std(dr_method_results, axis=2)

    for idx, shift in enumerate(shifts):
        errorfill(np.array(samples), mean_p_vals[:, idx], std_p_vals[:, idx], fmt=linestyles[idx]+markers[dr], color=colorscale(colors[dr],brightness[idx]), label="%s" % shift.replace('_', '\\_'))
    plt.xlabel('Number of samples from test')
    plt.ylabel('p-value')
    plt.axhline(y=sign_level, color='k')
    plt.savefig("%s/%s_conf_noleg.pdf" % (path, DimensionalityReduction(dr).name), bbox_inches='tight')
    plt.legend()
    plt.savefig("%s/%s_conf.pdf" % (path, DimensionalityReduction(dr).name), bbox_inches='tight')
    plt.clf()

np.save("%s/samples_shifts_rands_dr_tech.npy" % (path), samples_shifts_rands_dr_tech)
