from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, auc
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
import numpy as np
import time
import gc
import pandas as pd
import random
from joblib import Parallel, delayed

num_cores = 16


def split_data(data, test_size=0.25, random_state=0, y_column='insider',
               shuffle=True,
               x_rm_cols=('user', 'day', 'week', 'starttime', 'endtime', 'sessionid',
                          'timeind', 'Unnamed: 0', 'insider'),
               dname='r4.2', normalization='StandardScaler',
               rm_empty_cols=True, by_user=False, by_user_time=False,
               by_user_time_trainper=0.5, limit_ntrain_user=0):
    """
    split data to train and test, can get data by user, seq or random, with normalization builtin
    """
    np.random.seed(random_state)
    random.seed(random_state)

    x_cols = [i for i in data.columns if i not in x_rm_cols]
    if rm_empty_cols:
        x_cols = [i for i in x_cols if len(set(data[i])) > 1]

    infocols = list(set(data.columns) - set(x_cols))

    # output a dict
    out = {}

    # normalization
    if normalization == 'StandardScaler':
        sc = StandardScaler()
    elif normalization == 'MinMaxScaler':
        sc = MinMaxScaler()
    elif normalization == 'MaxAbsScaler':
        sc = MaxAbsScaler()
    else:
        sc = None
    out['sc'] = sc

    # split data randomly by instance
    if not by_user and not by_user_time:
        x = data[x_cols].values
        y_org = data[y_column].values

        y = y_org.copy()
        if 'r6' in dname:
            y[y != 0] = 1

        x_train, x_test, y_train, y_test = train_test_split(x, y_org, test_size=test_size, shuffle=shuffle)

        if 'sc' in out and out['sc'] is not None:
            x_train = sc.fit_transform(x_train)
            out['sc'] = sc
            if test_size > 0: x_test = sc.transform(x_test)

    # split data by user
    elif by_user:
        test_users, train_users = [], []
        for i in [j for j in list(set(data['insider'])) if j != 0]:
            uli = list(set(data[data['insider'] == i]['user']))
            random.shuffle(uli)
            ind_i = int(np.ceil(test_size * len(uli)))
            test_users += uli[:ind_i]
            train_users += uli[ind_i:]

        normal_users = list(set(data['user']) - set(data[data['insider'] != 0]['user']))
        random.shuffle(normal_users)
        if limit_ntrain_user > 0:
            normal_ind = limit_ntrain_user - len(train_users)
        else:
            normal_ind = int(np.ceil((1 - test_size) * len(normal_users)))

        train_users += normal_users[: normal_ind]
        test_users += normal_users[normal_ind:]
        x_train = data[data['user'].isin(train_users)][x_cols].values
        x_test = data[data['user'].isin(test_users)][x_cols].values
        y_train = data[data['user'].isin(train_users)][y_column].values
        y_test = data[data['user'].isin(test_users)][y_column].values

        out['train_info'] = data[data['user'].isin(train_users)][infocols]
        out['test_info'] = data[data['user'].isin(test_users)][infocols]

        out['train_users'] = train_users
        if test_size > 0 or limit_ntrain_user > 0:
            out['test_users'] = test_users

        if 'sc' in out and out['sc'] is not None:
            x_train = sc.fit_transform(x_train)
            out['sc'] = sc
            if test_size > 0 or (limit_ntrain_user > 0 and limit_ntrain_user < len(set(data['user']))):
                x_test = sc.transform(x_test)

    # split by user and time
    elif by_user_time:
        train_week_max = by_user_time_trainper * max(data['week'])
        train_insiders = set(data[(data['week'] <= train_week_max) & (data['insider'] != 0)]['user'])
        users_set_later_weeks = set(data[data['week'] > train_week_max]['user'])

        first_part = data[data['week'] <= train_week_max]
        second_part = data[data['week'] > train_week_max]

        first_part_split = split_data(first_part, random_state=random_state, test_size=0,
                                      dname=dname, normalization=normalization,
                                      by_user=True, by_user_time=False,
                                      limit_ntrain_user=limit_ntrain_user,
                                      )

        x_train = first_part_split['x_train']
        y_train = first_part_split['y_train']
        x_cols = first_part_split['x_cols']

        out['train_info'] = first_part_split['train_info']
        out['other_trainweeks_users_info'] = first_part_split['test_info']

        if 'sc' in first_part_split and first_part_split['sc'] is not None:
            out['sc'] = first_part_split['sc']

        out['x_other_trainweeks_users'] = first_part_split['x_test']
        out['y_other_trainweeks_users'] = first_part_split['y_test']
        out['y_bin_other_trainweeks_users'] = first_part_split['y_test_bin']
        out['other_trainweeks_users'] = first_part_split['test_users']  # users in first half but not in train

        real_train_users = set(first_part_split['train_users'])
        real_train_insiders = train_insiders.intersection(real_train_users)
        test_users = list(users_set_later_weeks - real_train_insiders)
        x_test = second_part[second_part['user'].isin(test_users)][x_cols].values
        y_test = second_part[second_part['user'].isin(test_users)][y_column].values
        out['test_info'] = second_part[second_part['user'].isin(test_users)][infocols]
        if ('sc' in out) and (out['sc'] is not None) and (by_user_time_trainper < 1):
            x_test = out['sc'].transform(x_test)

        out['train_users'] = first_part_split['train_users']
        out['test_users'] = test_users

    # get binary data
    y_train_bin = y_train.copy()
    y_train_bin[y_train_bin != 0] = 1

    out['x_train'] = x_train
    out['y_train'] = y_train
    out['y_train_bin'] = y_train_bin
    out['x_cols'] = x_cols
    out['info_cols'] = infocols

    out['test_size'] = test_size

    if test_size > 0 or (by_user_time and by_user_time_trainper < 1) or limit_ntrain_user > 0:
        y_test_bin = y_test.copy()
        y_test_bin[y_test_bin != 0] = 1
        out['x_test'] = x_test
        out['y_test'] = y_test
        out['y_test_bin'] = y_test_bin

    return out


def get_result_one_user(u, pred_all, datainfo):
    res_u = {}
    u_labels = datainfo[datainfo['user'] == u]['insider'].values
    utype = list(set(u_labels))
    if np.any(u_labels != 0) and len(set(u_labels)) > 1:
        utype.remove(0.0)
    res_u['type'] = utype[0]
    u_idx = np.where(datainfo['user'] == u)[0]
    res_u['data_idxs'] = u_idx
    pred = pred_all[u_idx]
    if len(np.where(u_labels == 0)[0]) > 0:
        res_u['norm_per'] = len(np.where(pred[u_labels == 0] == 0)[0]) / len(np.where(u_labels == 0)[0])
    if utype[0] != 0:
        res_u['mal_per'] = len(np.where(pred[u_labels != 0] != 0)[0]) / len(np.where(u_labels != 0)[0])
    res_u['norm_bin'] = int(np.any(pred[u_labels == 0] != 0))
    if utype[0] != 0:
        res_u['mal_bin'] = int(np.any(pred[u_labels != 0] != 0))
    return res_u


def get_result_by_users(users, user_list=None, pred_all=None, datainfo=None):
    out = {}
    cms = {}
    out[users] = {}
    # out_users = [get_result_one_user(u, pred_all, datainfo, old_res, label_all) for u in user_list]
    out_users = Parallel(n_jobs=num_cores)(delayed(get_result_one_user)(u, pred_all, datainfo)
                                           for u in user_list)
    users_true_label = []
    users_pred_label = []
    for i, u in enumerate(user_list):
        out[users][u] = out_users[i]
        users_true_label.append(out[users][u]['type'])
        if out[users][u]['type'] == 0:
            users_pred_label.append(out[users][u]['norm_bin'])
        else:
            users_pred_label.append(out[users][u]['mal_bin'])

    out[users]['true_label'] = users_true_label
    out[users]['pred_label'] = users_pred_label
    cms[users] = confusion_matrix(users_true_label, users_pred_label)
    return out, cms


def do_classification(clf, x_train, y_train, x_test, y_test, y_org=None, by_user=False,
                      split_output=None):
    '''
    train classification and get results
    '''
    st = time.time()
    clf.fit(x_train, y_train)
    train_time = time.time() - st

    cms_train = {}
    cms_test = {}

    st = time.time()
    y_train_hat = clf.predict(x_train)
    y_train_proba = clf.predict_proba(x_train)
    pred_time = time.time() - st
    cms_train['bin'] = confusion_matrix(y_train, y_train_hat)

    st = time.time()
    y_test_hat = clf.predict(x_test)
    y_test_proba = clf.predict_proba(x_test)
    test_pred_time = time.time() - st
    cms_test['bin'] = confusion_matrix(y_test, y_test_hat)

    test_org_labels = y_test
    train_org_labels = y_train
    if y_org is not None:
        cms_train['org'] = confusion_matrix(y_org['train'], y_train_hat)
        train_org_labels = y_org['train']
        cms_test['org'] = confusion_matrix(y_org['test'], y_test_hat)
        test_org_labels = y_org['test']

    userres_train = {}
    userres_test = {}
    if by_user:
        uout, ucm = get_result_by_users('train_users', split_output['train_users'], pred_all=y_train_hat,
                                        datainfo=split_output['train_info'])

        userres_train.update(uout)
        cms_train.update(ucm)
        uout, ucm = get_result_by_users('test_users', split_output['test_users'], pred_all=y_test_hat,
                                        datainfo=split_output['test_info'])
        userres_test.update(uout)
        cms_test.update(ucm)


    return clf, {'by_user': userres_train, 'cms': cms_train, 'train_time': train_time, 'pred_time': pred_time,
                 'org_labels': train_org_labels,
                 'pred_bin': y_train_hat, 'pred_proba': y_train_proba}, \
           {'by_user': userres_test, 'cms': cms_test, 'pred_time': test_pred_time, 'org_labels': test_org_labels,
            'pred_bin': y_test_hat,
            'pred_proba': y_test_proba}


def user_auc_roc(ures, users):
    nnu = len(users) - sum(ures[:, 0])
    nmu = sum(ures[:, 0])
    ufpr = np.sum(ures[np.where(ures[:, 0] == 0)[0], 1:], axis=0) / nnu
    utpr = np.sum(ures[np.where(ures[:, 0] == 1)[0], 1:], axis=0) / nmu
    uauc = auc(ufpr, utpr)
    return utpr, ufpr, uauc


def user_auc_roc2(u_idxs, y_bin, y_predprob, thresholds):
    ures = np.zeros((1, len(thresholds)))
    u_y = y_bin[u_idxs]
    u_yprob = y_predprob[u_idxs]
    for ii in range(len(thresholds[1:])):
        ures[0, ii + 1] = int(np.any(u_yprob >= thresholds[ii + 1]))
    if np.any(u_y != 0):
        ures[0, 0] = 1
    return ures


def roc_auc_calc(rw, algs=('RF', 'XGB'), nrun=20, dtype=None, data=None, res_names=['test_in']):

    allres = []
    fpri = np.linspace(0, 1, 1000)  # interpolation

    for i in range(nrun):
        for alg in algs:
            gc.collect()
            if 'all' in res_names:
                res_names = [td for td in rw[i][alg].keys() if td not in ['clf'] and 'thres' not in td]

            for resname in res_names:
                resname2 = 'train_users' if resname == 'train' else 'test_users'
                list_users = sorted([u for u in rw[i][alg][resname]['by_user'][resname2].keys() if type(u) != str])
                y_predprob = rw[i][alg][resname]['pred_proba'][:, 1]
                y_org = rw[i][alg][resname]['org_labels']
                y_bin = np.array(y_org > 0).astype(int)
                fpr, tpr, thresholds = roc_curve(y_bin, y_predprob)
                tpri = np.interp(fpri, fpr, tpr)

                if len(set(y_bin)) > 1:
                    aucsc = roc_auc_score(y_bin, y_predprob)
                    ures = Parallel(n_jobs=num_cores)(
                        delayed(user_auc_roc2)(rw[i][alg][resname]['by_user'][resname2][u]['data_idxs'], y_bin,
                                               y_predprob, thresholds) for u in list_users)

                    utpr, ufpr, uauc = user_auc_roc(np.vstack(ures), list_users)
                    utpri = np.interp(fpri, ufpr, utpr)
                else:
                    aucsc, ufpr, utpr, uauc, tpri, utpri = None, None, None, None, None, None

                allres.append([i, alg, resname, fpr, tpr, thresholds, aucsc, ufpr, utpr, uauc, fpri, tpri, utpri])

    res = pd.DataFrame(
        columns=['run', 'alg', 'test_on', 'fpr', 'tpr', 'threshold', 'auc', 'ufpr', 'utpr', 'uauc', 'fpri', 'tpri',
                 'utpri'], data=allres)
    if dtype is not None: res['dtype'] = dtype
    res['data'] = data
    return res


def get_cert_roc(r, a, dtype, test_on='test_in', user=True):
    fprs = r[(r['test_on'] == test_on) & (r['alg'] == a) & (r['dtype'] == dtype)]['fpri'].values
    if user:
        tprs = r[(r['test_on'] == test_on) & (r['alg'] == a) & (r['dtype'] == dtype)]['utpri'].values
        aucs = r[(r['test_on'] == test_on) & (r['alg'] == a) & (r['dtype'] == dtype)]['uauc'].values
    else:
        tprs = r[(r['test_on'] == test_on) & (r['alg'] == a) & (r['dtype'] == dtype)]['tpri'].values
        aucs = r[(r['test_on'] == test_on) & (r['alg'] == a) & (r['dtype'] == dtype)]['auc'].values

    mean_fpr = np.concatenate(([0], np.mean(fprs, axis=0)))
    mean_tpr = np.concatenate(([0], np.mean(tprs, axis=0)))

    std_auc = np.std(aucs)
    mean_auc = auc(mean_fpr, mean_tpr)

    std_tpr = np.concatenate(([0], np.std(tprs, axis=0)))
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

    return mean_fpr, mean_tpr, tprs_upper, tprs_lower, mean_auc, std_auc