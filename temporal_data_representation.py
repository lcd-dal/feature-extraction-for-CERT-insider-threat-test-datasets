# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import multiprocessing
from scipy.stats import percentileofscore
import argparse

def concat_combination(data, window_size = 3, dname = 'cert'):
    
    if dname == 'cert':
        info_cols = ['sessionid','day','week',"starttime", "endtime",
                     'user', 'project', 'role', 'b_unit', 'f_unit', 'dept', 'team', 'ITAdmin', 
                     'O', 'C', 'E', 'A', 'N', 'insider']
        
    combining_features = [ f for f in data.columns if f not in info_cols]
    info_features = [f for f in data.columns if f in info_cols]
    
    data_info = data[info_features].values
    
    data_combining_features = data[combining_features].values
    useridx = data['user'].values
    
    userset = set(data['user'])

    cols = []
    for shiftrange in range(window_size-1,0,-1):
        cols += [str(-shiftrange) + '_' + f for f in combining_features]
    cols += combining_features + info_features
    
    combined_data = []
    for u in userset:
        data_cf_u = data_combining_features[useridx == u, ]
        
        data_cf_u_shifted = []
        for shiftrange in range(window_size-1,0,-1):
            data_cf_u_shifted.append(np.roll(data_cf_u, shiftrange, axis = 0))
        
        data_cf_u_shifted.append(data_cf_u)
        data_cf_u_shifted.append(data_info[useridx==u, ])
        
        combined_data.append(np.hstack(data_cf_u_shifted)[window_size:,])
    
    combined_data = pd.DataFrame(np.vstack(combined_data), columns=cols)
    
    return combined_data


def subtract_combination_uworker(u, alluserdict, dtype, calc_type, window_size, udayidx, udata, uinfo, uorg):
    if u%200==0: 
        print(u)

    data_out = []
     
    if dtype in ['day', 'week']:
        
        for i in range(len(udayidx)):
            t = udayidx[i]
            if dtype in ['day','week']: min_idx = min(udayidx)+window_size
            
            if t>=min_idx:
                if calc_type == 'meandiff':
                    prevdata = udata[(udayidx > t - 1 - window_size) & (udayidx <= t-1),]
                    if len(prevdata) < 1: continue 
                    window_mean = np.mean(prevdata, axis = 0)
                    data_out.append(np.concatenate((udata[i] - window_mean, uorg[i,:], uinfo[i,:])))
                   
                if calc_type == 'meddiff':
                    prevdata = udata[(udayidx > t - 1 - window_size) & (udayidx <= t-1),]
                    if len(prevdata) < 1: continue 
                    window_med = np.median(prevdata, axis = 0)
                    data_out.append(np.concatenate((udata[i] - window_med, uorg[i,:], uinfo[i,:])))
                elif calc_type == 'percentile':
                    window = udata[(udayidx > t - 1 - window_size) & (udayidx <= t-1),]
                    if window.shape[0] < 1: continue
                    percentile_i = [percentileofscore(window[:,j], udata[i,j], 'mean') - 50 for j in range(window.shape[1])]
                    data_out.append(np.concatenate((percentile_i , uorg[i,:], uinfo[i,:])))
                    
    if len(data_out) > 0: alluserdict[u] = np.vstack(data_out)

def subtract_percentile_combination(data, dtype, calc_type = 'percentile', window_size = 7, dname = 'cert', parallel = True):
    '''
    Combine data to generate different temporal representations
    window_size: window size by days (for CERT data)
    '''
    if dname == 'cert':
        info_cols = ['sessionid','day','week',"starttime", "endtime", 
                     'user', 'project', 'role', 'b_unit', 'f_unit', 'dept', 'team', 'ITAdmin', 
                     'O', 'C', 'E', 'A', 'N', 'insider','subs_ind']
        keep_org_cols = ["pc", "isworkhour", "isafterhour", "isweekday", "isweekend", "isweekendafterhour", "n_days", 
                         "duration", "n_concurrent_sessions", "start_with", "end_with", "ses_start", "ses_end"]
        
    combining_features = [ f for f in data.columns if f not in info_cols]
    info_features = [f for f in data.columns if f in info_cols] 
    keep_org_features = [f for f in data.columns if f in keep_org_cols]
    
    data_info = data[info_features].values
    data_org = data[keep_org_features].values
    data_combining_features = data[combining_features].values
    useridx = data['user'].values
    if dtype in ['day']: dayidx = data['day'].values
    if dname == 'cert': weekidx = data['week'].values
    
    userset = set(data['user'])
    
    if dtype == 'week': 
        window_size = np.floor(window_size/7)
        idx = weekidx
    elif dtype in ['day']: idx = dayidx

    if parallel:
        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        jobs = []
        for u in userset:
            udayidx = idx[useridx==u]
            udata = data_combining_features[useridx==u, ]
            uinfo = data_info[useridx==u, ]
            uorg = data_org[useridx==u, ]
            p = multiprocessing.Process(target=subtract_combination_uworker, args=(u, return_dict, dtype, calc_type,
                                                                                    window_size, udayidx,
                                                                                    udata, uinfo, uorg))
            jobs.append(p)
            p.start()
    
        for proc in jobs:
            proc.join()
    else:
        return_dict = {}
        for u in userset:
            udayidx = idx[useridx==u]
            udata = data_combining_features[useridx==u, ]
            uinfo = data_info[useridx==u, ]
            uorg = data_org[useridx==u, ]
            subtract_combination_uworker(u, return_dict, dtype, calc_type,
                                        window_size, udayidx,
                                        udata, uinfo, uorg)

    combined_data = pd.DataFrame(np.vstack([return_dict[ri] for ri in return_dict.keys()]), columns=combining_features+['org_'+f for f in keep_org_features] + info_features)
    
    return combined_data


if __name__ == "__main__":    
    parser=argparse.ArgumentParser()
    parser.add_argument('--representation', help='Data representation to extract (concat, percentile, meandiff, mediandiff). Default: percentile', 
                        type= str, default = 'percentile')
    parser.add_argument('--file_input', help='CERT input file name. Default: week-r5.2.csv.gz', type= str, default= 'week-r5.2.csv.gz')  
    parser.add_argument('--window_size', help='Window size for percentile or mean/median difference representation. Default: 30', 
                        type = int, default=30)
    parser.add_argument('--num_concat', help='Number of data points for concatenation. Default: 3', 
                        type = int, default=3)
    args=parser.parse_args()    
    
    print('If "too many opened files", or "ForkAwareLocal" error, run ulimit command, e.g. "$ulimit -n 10000" to increase the limit first')
    if args.representation == 'all':
        reps = ['concat', 'percentile','meandiff','meddiff']
    elif args.representation in ['concat', 'percentile','meandiff','meddiff']:
        reps = [args.representation]
        
    fileName = (args.file_input).replace('.csv','').replace('.gz','')
    if 'day' in fileName:
        data_type = 'day'
    elif 'week' in fileName:
        data_type = 'week'
    s = pd.read_csv(f'{args.file_input}')
    
    for rep in reps:
        if rep in ['percentile','meandiff','meddiff']:
            s1 = subtract_percentile_combination(s, data_type, calc_type = rep, window_size = args.window_size, dname='cert')
            s1.to_pickle(f'{fileName}-{rep}{args.window_size}.pkl')
        else:
            s1 = concat_combination(s, window_size = args.num_concat, dname = 'cert')
            s1.to_pickle(f'{fileName}-{rep}{args.num_concat}.pkl')
    
