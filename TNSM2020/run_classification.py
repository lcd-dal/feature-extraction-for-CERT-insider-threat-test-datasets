from copy import deepcopy
import pickle
import gc
import pandas as pd
import time
import clf_helpers
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import warnings, sklearn

warnings.filterwarnings("ignore", category=sklearn.exceptions.DataConversionWarning)
n_cores = 16


def run_exp_onealg(run, slw, x, clf_name, classifiers, res_by_user, st):
    print(clf_name)
    clf_copy = deepcopy(classifiers[clf_name])
    if hasattr(clf_copy, 'random_state'):
        clf_copy.set_params(**{'random_state': run})

    clf_res = clf_helpers.do_classification(clf_copy, x['train'], slw['y_train_bin'],
                                        x['test'], slw['y_test_bin'],
                                        y_org={'train': slw['y_train'], 'test': slw['y_test']},
                                        by_user=res_by_user, split_output=slw)
    res = {'train': clf_res[1], 'test_in': clf_res[2]}
    print('Training time: ', res['train']['train_time'])
    print('Train confusion matrices: ', res['train']['cms'])
    print('Test confusion matrices: ', res['test_in']['cms'])
    gc.collect()
    print(run, 'Done training & res,', (time.time() - st) // 1)
    return res


def run_exp_onerun(run, classifiers, data_in, test_size, res_by_user, shuffle,
                   normalization, by_user_time=False, by_user_time_trainper=0.5, limit_ntrain_user=0):
    st = time.time()
    slw = clf_helpers.split_data(data_in['data'], test_size=test_size, shuffle=shuffle, random_state=run,
                                 normalization=normalization, dname=data_in['name'],
                                 by_user_time=by_user_time, by_user_time_trainper=by_user_time_trainper,
                                 limit_ntrain_user=limit_ntrain_user)
    print('\n', run, 'Done splitting,', (time.time() - st) // 1)

    res_clf = {'scaler': slw['sc']}
    x_cols = list(slw['x_cols'])
    x = {'train': slw['x_train'], 'test': slw['x_test']}

    for clf_name in classifiers:
        res = run_exp_onealg(run, slw, x, clf_name, classifiers, res_by_user, st)
        res_clf[clf_name] = res

    return res_clf, x_cols


def run_experiment(n_run, classifiers, data_in, test_size=0.5, res_by_user=True, shuffle=True,
                   normalization="StandardScaler", by_user_time=True, by_user_time_trainper=0.5,
                   limit_ntrain_user=0):

    all_res = {'exp_setting': {'y_col': 'insider', 'train_dname': data_in['name'],
                               'shuffle': shuffle, 'norm': normalization, 'res_by_user': res_by_user}}
    all_res['exp_setting']['classifiers'] = list(classifiers.keys())
    all_res['exp_setting']['n_run'] = n_run
    all_res['exp_setting']['in_test_size'] = test_size
    all_res['exp_setting']['by_user_time_trainper'] = by_user_time_trainper
    all_res['exp_setting']['limit_ntrain_user'] = limit_ntrain_user

    for run in range(n_run):
        res_clf, x_cols = run_exp_onerun(run, classifiers, data_in, test_size,
                                         res_by_user=res_by_user, by_user_time=by_user_time,
                                         by_user_time_trainper=by_user_time_trainper,
                                         limit_ntrain_user=limit_ntrain_user,
                                         shuffle=shuffle,
                                         normalization=normalization,
                                         )
        gc.collect()
        all_res[run] = res_clf
    return all_res


def load_data(dname, dtype, datafolder='data'):
    name = datafolder + '/' + dtype + dname + '.csv.gz'
    return pd.read_csv(name)


def run_exp(nrun, dname, dtype, mode, ttype=None, limit_ntrain_user=None, train_week_per=None, test_per=0.5, algs=None,
            load_params=True, scaler='StandardScaler', savefolder='res'):

    print('\n----------------\n%s %s' % (dname, dtype), '\n----------------\n')

    clfs = {'LR': LogisticRegression(solver='lbfgs', n_jobs=n_cores),
            'MLP': MLPClassifier(solver='adam'),
            'RF': RandomForestClassifier(n_jobs=n_cores),
            'XGB': XGBClassifier(n_jobs=n_cores),
            }

    if algs is not None:
        clfs = {k:clfs[k] for k in algs}

    if load_params:
        with open('params.pkl', 'rb') as f:
            loaded_params = pickle.load(f)
            for c in clfs:
                if c != 'LR':
                    clfs[c].set_params(**loaded_params[c][dtype])

    data_in = {'name': dname, 'data': load_data(dname, dtype)}

    if mode == 'by_user_time':
        res = run_experiment(nrun, clfs, data_in, by_user_time=True,
                             by_user_time_trainper=train_week_per,
                             limit_ntrain_user=limit_ntrain_user,
                             res_by_user=True, normalization=scaler)
    elif mode == 'randomsplit':
        res = run_experiment(nrun, clfs, data_in, by_user_time=False,
                             test_size=test_per,
                             res_by_user=False, normalization=scaler)

    savefile = '%s/%s-%s-%s-%s-%s' % (savefolder, dname, dtype, ttype, mode, '_'.join(algs)) + '.pickle'
    with open(savefile, 'wb') as handle:
        pickle.dump(res, handle, protocol=4)
    return res


if __name__ == "__main__":
    algs = ['RF']
    nrun = 2

    dname = 'r5.2'
    dtypes = ['week']
    mode = 'by_user_time'

    for dtype in dtypes:
        res = run_exp(nrun, dname, dtype, algs=algs, mode=mode, limit_ntrain_user=400, train_week_per=0.5,
                      load_params=True)
        if mode == 'randomsplit': continue

        res = clf_helpers.roc_auc_calc(res, algs=algs, nrun=nrun, dtype=dtype, data=dname)

        colors = ['r', 'g', 'blue', 'orange']
        for user in [True, False]:
            plt.figure()
            restype = 'user' if user else 'org'
            for i, alg in enumerate(algs):
                tmp = clf_helpers.get_cert_roc(res, alg, dtype, 'test_in', user=user)
                plt.plot(tmp[0], tmp[1], label=f'{alg}, AUC = {tmp[4]:.3f}', color=colors[i])
                plt.fill_between(tmp[0], tmp[3], tmp[2], color=colors[i], alpha=.1, label=None)
            plt.legend()
            plt.savefig(f'ROC_{dtype}_{restype}.jpg')
