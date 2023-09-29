#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: lcd
"""
import os, sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
import time
import subprocess
from joblib import Parallel, delayed

def time_convert(inp, mode, real_sd = '2010-01-02', sd_monday= "2009-12-28"):
    if mode == 'e2t':
        return datetime.fromtimestamp(inp).strftime('%m/%d/%Y %H:%M:%S')
    elif mode == 't2e':
        return datetime.strptime(inp, '%m/%d/%Y %H:%M:%S').strftime('%s')
    elif mode == 't2dt':
        return datetime.strptime(inp, '%m/%d/%Y %H:%M:%S')
    elif mode == 't2date':
        return datetime.strptime(inp, '%m/%d/%Y %H:%M:%S').strftime("%Y-%m-%d")
    elif mode == 'dt2t':
        return inp.strftime('%m/%d/%Y %H:%M:%S')
    elif mode == 'dt2W':
        return int(inp.strftime('%W'))
    elif mode == 'dt2d':
        return inp.strftime('%m/%d/%Y %H:%M:%S')
    elif mode == 'dt2date':
        return inp.strftime("%Y-%m-%d")
    elif mode =='dt2dn': #datetime to day number
        startdate = datetime.strptime(sd_monday,'%Y-%m-%d')
        return (inp - startdate).days
    elif mode =='dn2epoch': #datenum to epoch
        dt = datetime.strptime(sd_monday,'%Y-%m-%d') + timedelta(days=inp)
        return int(dt.timestamp())
    elif mode =='dt2wn': #datetime to week number
        startdate = datetime.strptime(real_sd,'%Y-%m-%d')
        return (inp - startdate).days//7
    elif mode =='t2wn': #datetime to week number
        startdate = datetime.strptime(real_sd,'%Y-%m-%d')
        return (datetime.strptime(inp, '%m/%d/%Y %H:%M:%S') - startdate).days//7
    elif mode == 'dt2wd':
        return int(inp.strftime("%w"))
    elif mode == 'm2dt':
        return datetime.strptime(inp, "%Y-%m")
    elif mode == 'datetoweekday':
        return int(datetime.strptime(inp,"%Y-%m-%d").strftime('%w'))
    elif mode == 'datetoweeknum':
        w0 = datetime.strptime(sd_monday,"%Y-%m-%d")
        return int((datetime.strptime(inp,"%Y-%m-%d") - w0).days / 7)
    elif mode == 'weeknumtodate':
        startday = datetime.strptime(sd_monday,"%Y-%m-%d")
        return startday+timedelta(weeks = inp)
    
def add_action_thisweek(act, columns, lines, act_handles, week_index, stop, firstdate, dname = 'r5.2'):
    thisweek_act = []
    while True:
        if not lines[act]: 
            stop[act] = 1
            break
        if dname in ['r6.1','r6.2'] and act in ['email', 'file','http'] and '"' in lines[act]:
            tmp = lines[act]
            firstpart = tmp[:tmp.find('"')-1]
            content = tmp[tmp.find('"')+1:-1]
            tmp = firstpart.split(',') + [content]
        else:
            tmp = lines[act].split(',')
        if time_convert(tmp[1], 't2wn', real_sd= firstdate) == week_index:
            thisweek_act.append(tmp)
        else:
            break
        lines[act] = act_handles[act].readline()
    df = pd.DataFrame(thisweek_act, columns=columns)
    df['type']= act
    df.index = df['id']
    df.drop('id', axis=1, inplace=True)

    return df

def combine_by_timerange_pandas(dname = 'r4.2'):
    allacts =  ['device','email','file', 'http','logon']
    firstline = str(subprocess.check_output(['head', '-2', 'http.csv'])).split('\\n')[1]
    firstdate = time_convert(firstline.split(',')[1],'t2dt')
    firstdate = firstdate - timedelta(int(firstdate.strftime("%w")))
    firstdate = time_convert(firstdate, 'dt2date')
    week_index = 0
    act_handles = {}
    lines = {}
    stop = {}
    for act in allacts:
        act_handles[act] = open(act+'.csv','r')
        next(act_handles[act],None) #skip header row
        lines[act] = act_handles[act].readline()
        stop[act] = 0 # store stop value indicating if all of the file has been read
    while sum(stop.values()) < 5:
        thisweekdf = pd.DataFrame()
        for act in allacts:
            if 'email' == act:
                if dname in ['r4.1','r4.2']:
                    columns = ['id', 'date', 'user', 'pc', 'to', 'cc', 'bcc', 'from', 'size', '#att', 'content']
                if dname in ['r6.1','r6.2','r5.2','r5.1']:
                    columns = ['id', 'date', 'user', 'pc', 'to', 'cc', 'bcc', 'from', 'activity', 'size', 'att', 'content']     
            elif 'logon' == act:
                columns = ['id', 'date', 'user', 'pc', 'activity']
            elif 'device' == act:
                if dname in ['r4.1','r4.2']:
                    columns = ['id', 'date', 'user', 'pc', 'activity']
                if dname in ['r5.1','r5.2','r6.2','r6.1']:
                    columns = ['id', 'date', 'user', 'pc', 'content', 'activity']
            elif 'http' == act:
                if dname in ['r6.1','r6.2']: columns = ['id', 'date', 'user', 'pc', 'url/fname', 'activity', 'content']
                if dname in ['r5.1','r5.2','r4.2','r4.1']: columns = ['id', 'date', 'user', 'pc', 'url/fname', 'content']
            elif 'file' == act:
                if dname in ['r4.1','r4.2']: columns = ['id', 'date', 'user', 'pc', 'url/fname', 'content']
                if dname in ['r5.2','r5.1','r6.2','r6.1']: columns = ['id', 'date', 'user', 'pc', 'url/fname','activity','to','from','content']
            
            df = add_action_thisweek(act, columns, lines, act_handles, week_index, stop, firstdate, dname=dname)
        thisweekdf = pd.concat([thisweekdf, df], ignore_index=True)        
        thisweekdf['date'] = thisweekdf['date'].apply(lambda x: datetime.strptime(x, "%m/%d/%Y %H:%M:%S"))
        thisweekdf.to_pickle("DataByWeek/"+str(week_index)+".pickle")
        week_index += 1

##############################################################################

def process_user_pc(upd, roles): #figure out  which PC belongs to which user
    upd['sharedpc'] = None
    upd['npc'] = upd['pcs'].apply(lambda x: len(x))
    upd.at[upd['npc']==1,'pc'] = upd[upd['npc']==1]['pcs'].apply(lambda x: x[0])
    multiuser_pcs = np.concatenate(upd[upd['npc']>1]['pcs'].values).tolist()
    set_multiuser_pc = list(set(multiuser_pcs))
    count = {}
    for pc in set_multiuser_pc:
        count[pc] = multiuser_pcs.count(pc)
    for u in upd[upd['npc']>1].index:
        sharedpc = upd.loc[u]['pcs']
        count_u_pc = [count[pc] for pc in upd.loc[u]['pcs']]
        the_pc = count_u_pc.index(min(count_u_pc))
        upd.at[u,'pc'] = sharedpc[the_pc]
        if roles.loc[u] != 'ITAdmin':
            sharedpc.remove(sharedpc[the_pc])
            upd.at[u,'sharedpc']= sharedpc
    return upd

def getuserlist(dname = 'r4.2', psycho = True):
    allfiles =  ['LDAP/'+f1 for f1 in os.listdir('LDAP') if os.path.isfile('LDAP/'+f1)]
    alluser = {}
    alreadyFired = []
    
    for file in allfiles:
        af = (pd.read_csv(file,delimiter=',')).values
        employeesThisMonth = []    
        for i in range(len(af)):
            employeesThisMonth.append(af[i][1])
            if af[i][1] not in alluser:
                alluser[af[i][1]] = af[i][0:1].tolist() + af[i][2:].tolist() + [file.split('.')[0] , np.nan]

        firedEmployees = list(set(alluser.keys()) - set(alreadyFired) - set(employeesThisMonth))
        alreadyFired = alreadyFired + firedEmployees
        for e in firedEmployees:
            alluser[e][-1] = file.split('.')[0]
    
    if psycho and os.path.isfile("psychometric.csv"):

        p_score = pd.read_csv("psychometric.csv",delimiter = ',').values
        for id in range(len(p_score)):
            alluser[p_score[id,1]] = alluser[p_score[id,1]]+ list(p_score[id,2:])
        df  = pd.DataFrame.from_dict(alluser, orient='index')
        if dname in ['r4.1','r4.2']:
            df.columns = ['uname', 'email', 'role', 'b_unit', 'f_unit', 'dept', 'team', 'sup','wstart', 'wend', 'O', 'C', 'E', 'A', 'N']
        elif dname in ['r5.2','r5.1','r6.2','r6.1']:
            df.columns = ['uname', 'email', 'role', 'project', 'b_unit', 'f_unit', 'dept', 'team', 'sup','wstart', 'wend', 'O', 'C', 'E', 'A', 'N']
    else:
        df  = pd.DataFrame.from_dict(alluser, orient='index')
        if dname in ['r4.1','r4.2']:
            df.columns = ['uname', 'email', 'role', 'b_unit', 'f_unit', 'dept', 'team', 'sup', 'wstart', 'wend']
        elif dname in ['r5.2','r5.1','r6.2','r6.1']:
            df.columns = ['uname', 'email', 'role', 'project', 'b_unit', 'f_unit', 'dept', 'team', 'sup', 'wstart', 'wend']

    df['pc'] = None
    for i in df.index:
        if type(df.loc[i]['sup']) == str:
            sup = df[df['uname'] == df.loc[i]['sup']].index[0]
        else:
            sup = None
        df.at[i,'sup'] = sup
        
    #read first 2 weeks to determine each user's PC
    w1 = pd.read_pickle("DataByWeek/1.pickle")
    w2 = pd.read_pickle("DataByWeek/2.pickle")
    user_pc_dict = pd.DataFrame(index=df.index)
    user_pc_dict['pcs'] = None  
  
    for u in df.index:
        pc = list(set(w1[w1['user']==u]['pc']) & set(w2[w2['user']==u]['pc']))
        user_pc_dict.at[u,'pcs'] = pc
    upd = process_user_pc(user_pc_dict, df['role'])
    df['pc'] = upd['pc']
    df['sharedpc'] = upd['sharedpc']
    return df

        
def get_mal_userdata(data = 'r4.2', usersdf = None):
    
    if not os.path.isdir('answers'):
        os.system('wget https://kilthub.cmu.edu/ndownloader/files/24857828 -O answers.tar.bz2')
        os.system('tar -xjvf answers.tar.bz2')
    
    listmaluser = pd.read_csv("answers/insiders.csv")
    listmaluser['dataset'] = listmaluser['dataset'].apply(lambda x: str(x))
    listmaluser = listmaluser[listmaluser['dataset']==data.replace("r","")]
    #for r6.2, new time in scenario 4 answer is incomplete.
    if data == 'r6.2': listmaluser.at[listmaluser['scenario']==4,'start'] = '02'+listmaluser[listmaluser['scenario']==4]['start']
    listmaluser[['start','end']] = listmaluser[['start','end']].applymap(lambda x: datetime.strptime(x, "%m/%d/%Y %H:%M:%S"))
    
    if type(usersdf) != pd.core.frame.DataFrame:
        usersdf = getuserlist(data)
    usersdf['malscene']=0
    usersdf['mstart'] = None
    usersdf['mend'] = None
    usersdf['malacts'] = None
    
    for i in listmaluser.index:
        usersdf.loc[listmaluser['user'][i], 'mstart'] = listmaluser['start'][i]
        usersdf.loc[listmaluser['user'][i], 'mend'] = listmaluser['end'][i]
        usersdf.loc[listmaluser['user'][i], 'malscene'] = listmaluser['scenario'][i]
        
        if data in ['r4.2', 'r5.2']:
            malacts = open(f"answers/r{listmaluser['dataset'][i]}-{listmaluser['scenario'][i]}/"+
                       listmaluser['details'][i],'r').read().strip().split("\n")
        else: #only 1 malicious user, no folder
            malacts = open("answers/"+ listmaluser['details'][i],'r').read().strip().split("\n")
        
        malacts = [x.split(',') for x in malacts]

        mal_users = np.array([x[3].strip('"') for x in malacts])
        mal_act_ids =  np.array([x[1].strip('"') for x in malacts])
        
        usersdf.at[listmaluser['user'][i], 'malacts'] = mal_act_ids[mal_users==listmaluser['user'][i]]
                    
    return usersdf

##############################################################################

def is_after_whour(dt): #Workhours assumed 7:30-17:30
    wday_start = datetime.strptime("7:30", "%H:%M").time()
    wday_end = datetime.strptime("17:30", "%H:%M").time()
    dt = dt.time()
    if dt < wday_start or dt > wday_end:
        return True
    return False
      
def is_weekend(dt):
    if dt.strftime("%w") in ['0', '6']:
        return True
    return False   
    
def email_process(act, data = 'r4.2', separate_send_receive = True):
    receivers = act['to'].split(';')
    if type(act['cc']) == str:
        receivers = receivers + act['cc'].split(";")
    if type(act['bcc']) == str:
        bccreceivers = act['bcc'].split(";")   
    else:
        bccreceivers = []
    exemail = False
    n_exdes = 0
    for i in receivers + bccreceivers:
        if 'dtaa.com' not in i:
            exemail = True
            n_exdes += 1

    n_des = len(receivers) + len(bccreceivers)
    Xemail = 1 if exemail else 0
    n_bccdes = len(bccreceivers)
    exbccmail = 0
    email_text_len = len(act['content'])
    email_text_nwords = act['content'].count(' ') + 1
    for i in bccreceivers:
        if 'dtaa.com' not in i:
            exbccmail = 1
            break

    if data in ['r5.1','r5.2','r6.1','r6.2']:
        send_mail = 1 if act['activity'] == 'Send' else 0
        receive_mail = 1 if act['activity'] in ['Receive','View'] else 0
        
        atts = act['att'].split(';')
        n_atts = len(atts)
        size_atts = 0
        att_types = [0,0,0,0,0,0]
        att_sizes = [0,0,0,0,0,0]
        for att in atts:
            if '.' in att:
                tmp = file_process(att, filetype='att')
                att_types = [sum(x) for x in zip(att_types,tmp[0])]
                att_sizes = [sum(x) for x in zip(att_sizes,tmp[1])]
                size_atts +=sum(tmp[1])
        return [send_mail, receive_mail, n_des, n_atts, Xemail, n_exdes, 
                n_bccdes, exbccmail, int(act['size']), email_text_len, 
                email_text_nwords] + att_types + att_sizes
    elif data in ['r4.1','r4.2']:
        return [n_des, int(act['#att']), Xemail, n_exdes, n_bccdes, exbccmail, 
                int(act['size']), email_text_len, email_text_nwords]
        
def http_process(act, data = 'r4.2'): 
    # basic features:
    url_len = len(act['url/fname'])
    url_depth = act['url/fname'].count('/')-2
    content_len = len(act['content'])
    content_nwords = act['content'].count(' ')+1
    
    domainname = re.findall("//(.*?)/", act['url/fname'])[0]
    domainname.replace("www.","")
    dn = domainname.split(".")
    if len(dn) > 2 and not any([x in domainname for x in ["google.com", '.co.uk', '.co.nz', 'live.com']]):
        domainname = ".".join(dn[-2:])

    # other 1, socnet 2, cloud 3, job 4, leak 5, hack 6
    if domainname in ['dropbox.com', 'drive.google.com', 'mega.co.nz', 'account.live.com']:
        r = 3
    elif domainname in ['wikileaks.org','freedom.press','theintercept.com']:
        r = 5
    elif domainname in ['facebook.com','twitter.com','plus.google.com','instagr.am','instagram.com',
                        'flickr.com','linkedin.com','reddit.com','about.com','youtube.com','pinterest.com',
                        'tumblr.com','quora.com','vine.co','match.com','t.co']:
        r = 2
    elif domainname in ['indeed.com','monster.com', 'careerbuilder.com','simplyhired.com']:
        r = 4
    
    elif ('job' in domainname and ('hunt' in domainname or 'search' in domainname)) \
    or ('aol.com' in domainname and ("recruit" in act['url/fname'] or "job" in act['url/fname'])):
        r = 4
    elif (domainname in ['webwatchernow.com','actionalert.com', 'relytec.com','refog.com','wellresearchedreviews.com',
                         'softactivity.com', 'spectorsoft.com','best-spy-soft.com']):
        r = 6
    elif ('keylog' in domainname):
        r = 6
    else:
        r = 1
    if data in ['r6.1','r6.2']:
        http_act_dict = {'www visit': 1, 'www download': 2, 'www upload': 3}
        http_act = http_act_dict.get(act['activity'].lower(), 0)
        return [r, url_len, url_depth, content_len, content_nwords, http_act]
    else:
        return [r, url_len, url_depth, content_len, content_nwords]
        
def file_process(act, complete_ul = None, data = 'r4.2', filetype = 'act'):
    if filetype == 'act':
        ftype = act['url/fname'].split(".")[1]
        disk = 1 if act['url/fname'][0] == 'C' else 0
        if act['url/fname'][0] == 'R': disk = 2
        file_depth = act['url/fname'].count('\\')
    elif filetype == 'att': #attachments
        tmp = act.split('.')[1]
        ftype = tmp[:tmp.find('(')]
        attsize = int(tmp[tmp.find("(")+1:tmp.find(")")])
        r = [[0,0,0,0,0,0], [0,0,0,0,0,0]]
        if ftype in ['zip','rar','7z']:
            ind = 1
        elif ftype in ['jpg', 'png', 'bmp']:
            ind = 2
        elif ftype in ['doc','docx', 'pdf']:
            ind = 3
        elif ftype in ['txt','cfg', 'rtf']:
            ind = 4
        elif ftype in ['exe', 'sh']:
            ind = 5
        else:
            ind = 0
        r[0][ind] = 1
        r[1][ind] = attsize
        return r

    fsize = len(act['content'])
    f_nwords = act['content'].count(' ')+1
    if ftype in ['zip','rar','7z']:
        r = 2
    elif ftype in ['jpg', 'png', 'bmp']:
        r = 3
    elif ftype in ['doc','docx', 'pdf']:
        r = 4
    elif ftype in ['txt','cfg','rtf']:
        r = 5
    elif ftype in ['exe', 'sh']:
        r = 6
    else:
        r = 1
    if data in ['r5.2','r5.1', 'r6.2','r6.1']:
        to_usb = 1 if act['to'] == 'True' else 0
        from_usb = 1 if act['from'] == 'True' else 0
        file_depth = act['url/fname'].count('\\')
        file_act_dict = {'file open': 1, 'file copy': 2, 'file write': 3, 'file delete': 4}
        if act['activity'].lower() not in file_act_dict: print(act['activity'].lower())
        file_act = file_act_dict.get(act['activity'].lower(), 0)
        return [r, fsize, f_nwords, disk, file_depth, file_act, to_usb, from_usb]
    elif data in ['r4.1','r4.2']:
        return [r, fsize, f_nwords, disk, file_depth]

def from_pc(act, ul):
    #code: 0,1,2,3:  own pc, sharedpc, other's pc, supervisor's pc
    user_pc = ul.loc[act['user']]['pc']
    act_pc = act['pc']
    if act_pc == user_pc:
        return (0, act_pc) #using normal PC
    elif ul.loc[act['user']]['sharedpc'] is not None and act_pc in ul.loc[act['user']]['sharedpc']:
        return (1, act_pc)
    elif ul.loc[act['user']]['sup'] is not None and act_pc == ul.loc[ul.loc[act['user']]['sup']]['pc']:
        return (3, act_pc)
    else:
        return (2, act_pc)
    
def process_week_num(week, users, userlist = 'all', data = 'r4.2'):

    user_dict = {idx: i for (i, idx) in enumerate(users.index)}        
    acts_week = pd.read_pickle("DataByWeek/"+str(week)+".pickle")
    start_week, end_week = min(acts_week.date), max(acts_week.date)
    acts_week.sort_values('date', ascending = True, inplace = True)
    n_cols = 45 if data in ['r5.2','r5.1'] else 46
    if data in ['r4.2','r4.1']: n_cols = 27
    u_week = np.zeros((len(acts_week), n_cols))
    pc_time = []
    if userlist == 'all':
        userlist = set(acts_week.user)
    
    #FOR EACH USER
    current_ind = 0
    for u in userlist:
        df_acts_u = acts_week[acts_week.user == u]
        mal_u = 0 #, stop_soon = 0, 0        
        if users.loc[u].malscene > 0:
            if start_week <= users.loc[u].mend and users.loc[u].mstart <= end_week:
                mal_u = users.loc[u].malscene
        
        list_uacts = df_acts_u.type.tolist() #all user's activities       
        list_activity = df_acts_u.activity.tolist()
        list_uacts = [list_activity[i].strip().lower() if (type(list_activity[i])==str and list_activity[i].strip() in ['Logon', 'Logoff', 'Connect', 'Disconnect']) \
                        else list_uacts[i] for i in range(len(list_uacts))]  
        uacts_mapping = {'logon':1, 'logoff':2, 'connect':3, 'disconnect':4, 'http':5,'email':6,'file':7}
        list_uacts_num = [uacts_mapping[x] for x in list_uacts]

        oneu_week = np.zeros((len(df_acts_u), n_cols))
        oneu_pc_time = []
        for i in range(len(df_acts_u)):
            pc, _ = from_pc(df_acts_u.iloc[i], users)
            if is_weekend(df_acts_u.iloc[i]['date']):
                if is_after_whour(df_acts_u.iloc[i]['date']):
                    act_time = 4
                else:
                    act_time = 3
            elif is_after_whour(df_acts_u.iloc[i]['date']):
                act_time = 2
            else:
                act_time = 1
            
            if data in ['r4.2','r4.1']:
                device_f = [0]
                file_f = [0, 0, 0, 0, 0]
                http_f = [0,0,0,0,0]
                email_f = [0]*9
            elif data in ['r5.2','r5.1','r6.2','r6.1']:
                device_f = [0,0]
                file_f = [0]*8
                http_f = [0,0,0,0,0]
                if data in ['r6.2','r6.1']:
                    http_f = [0,0,0,0,0,0]
                email_f = [0]*23
            
            if list_uacts[i] == 'file':
                file_f = file_process(df_acts_u.iloc[i], data = data)
            elif list_uacts[i] == 'email':
                email_f = email_process(df_acts_u.iloc[i], data = data)
            elif list_uacts[i] == 'http':
                http_f = http_process(df_acts_u.iloc[i], data=data)
            elif list_uacts[i] == 'connect':
                tmp = df_acts_u.iloc[i:]
                disconnect_acts = tmp[(tmp['activity'] == 'Disconnect\n') & \
                 (tmp['user'] == df_acts_u.iloc[i]['user']) & \
                 (tmp['pc'] == df_acts_u.iloc[i]['pc'])]
                
                connect_acts = tmp[(tmp['activity'] == 'Connect\n') & \
                 (tmp['user'] == df_acts_u.iloc[i]['user']) & \
                 (tmp['pc'] == df_acts_u.iloc[i]['pc'])]
                
                if len(disconnect_acts) > 0:
                    distime = disconnect_acts.iloc[0]['date']
                    if len(connect_acts) > 0 and connect_acts.iloc[0]['date'] < distime:
                        connect_dur = -1
                    else:
                        tmp_td = distime - df_acts_u.iloc[i]['date']
                        connect_dur = tmp_td.days*24*3600 + tmp_td.seconds
                else:
                    connect_dur = -1 # disconnect action not found!
                    
                if data in ['r5.2','r5.1','r6.2','r6.1']:
                    file_tree_len =  len(df_acts_u.iloc[i]['content'].split(';'))
                    device_f = [connect_dur, file_tree_len]
                else:
                    device_f = [connect_dur]
                
            is_mal_act = 0
            if mal_u > 0 and df_acts_u.index[i] in users.loc[u]['malacts']: is_mal_act = 1

            oneu_week[i,:] = [ user_dict[u], time_convert(df_acts_u.iloc[i]['date'], 'dt2dn'), list_uacts_num[i], pc, act_time] \
            + device_f + file_f + http_f + email_f + [is_mal_act, mal_u]

            oneu_pc_time.append([df_acts_u.index[i], df_acts_u.iloc[i]['pc'],df_acts_u.iloc[i]['date']])
        u_week[current_ind:current_ind+len(oneu_week),:] = oneu_week
        pc_time += oneu_pc_time
        current_ind += len(oneu_week)
    
    u_week = u_week[0:current_ind, :]
    col_names = ['user','day','act','pc','time']
    if data in ['r4.1','r4.2']:
        device_feature_names = ['usb_dur']
        file_feature_names = ['file_type', 'file_len', 'file_nwords', 'disk', 'file_depth']
        http_feature_names = ['http_type', 'url_len','url_depth', 'http_c_len', 'http_c_nwords']
        email_feature_names = ['n_des', 'n_atts', 'Xemail', 'n_exdes', 'n_bccdes', 'exbccmail', 'email_size', 'email_text_slen', 'email_text_nwords']
    elif data in ['r5.2','r5.1', 'r6.2','r6.1']:
        device_feature_names = ['usb_dur', 'file_tree_len']
        file_feature_names = ['file_type', 'file_len', 'file_nwords', 'disk', 'file_depth', 'file_act', 'to_usb', 'from_usb']
        http_feature_names = ['http_type', 'url_len','url_depth', 'http_c_len', 'http_c_nwords']
        if data in ['r6.2','r6.1']:
            http_feature_names = ['http_type', 'url_len','url_depth', 'http_c_len', 'http_c_nwords', 'http_act']
        email_feature_names = ['send_mail', 'receive_mail','n_des', 'n_atts', 'Xemail', 'n_exdes', 'n_bccdes', 'exbccmail', 'email_size', 'email_text_slen', 'email_text_nwords']
        email_feature_names += ['e_att_other', 'e_att_comp', 'e_att_pho', 'e_att_doc', 'e_att_txt', 'e_att_exe']
        email_feature_names += ['e_att_sother', 'e_att_scomp', 'e_att_spho', 'e_att_sdoc', 'e_att_stxt', 'e_att_sexe']     
        
    col_names = col_names + device_feature_names + file_feature_names+ http_feature_names + email_feature_names + ['mal_act','insider']#['stop_soon', 'mal_act','insider']
    df_u_week = pd.DataFrame(columns=['actid','pcid','time_stamp'] + col_names, index = np.arange(0,len(pc_time)))
    df_u_week[['actid','pcid','time_stamp']] = np.array(pc_time)
    
    df_u_week[col_names] = u_week
    df_u_week[col_names] = df_u_week[col_names].astype(int)
    df_u_week.to_pickle("NumDataByWeek/"+str(week)+"_num.pickle")

##############################################################################

# return sessions for each user in a week:
# sessions[sid] = [sessionid, pc, start_with, end_with, start time, end time,number_of_concurent_login, [action_indices]]
# start_with: in the beginning of a week, action start with log in or not (1, 2)
# end_with: log off, next log on same computer (1, 2)
def get_sessions(uw, first_sid = 0):
    sessions = {}
    open_sessions = {}
    sid = 0
    current_pc = uw.iloc[0]['pcid']
    start_time = uw.iloc[0]['time_stamp']
    if uw.iloc[0]['act'] == 1:
        open_sessions[current_pc] = [current_pc, 1, 0, start_time, start_time, 1, [uw.index[0]]]
    else:
        open_sessions[current_pc] = [current_pc, 2, 0, start_time, start_time, 1, [uw.index[0]]]

    for i in uw.index[1:]:
        current_pc = uw.loc[i]['pcid']
        if current_pc in open_sessions: # must be already a session with that pcid
            if uw.loc[i]['act'] == 2:
                open_sessions[current_pc][2] = 1
                open_sessions[current_pc][4] = uw.loc[i]['time_stamp']
                open_sessions[current_pc][6].append(i)
                sessions[sid] = [first_sid+sid] + open_sessions.pop(current_pc)
                sid +=1
            elif uw.loc[i]['act'] == 1:
                open_sessions[current_pc][2] = 2
                sessions[sid] = [first_sid+sid] + open_sessions.pop(current_pc)
                sid +=1
                #create a new open session
                open_sessions[current_pc] = [current_pc, 1, 0, uw.loc[i]['time_stamp'], uw.loc[i]['time_stamp'], 1, [i]]
                if len(open_sessions) > 1: #increase the concurent count for all sessions
                    for k in open_sessions:
                        open_sessions[k][5] +=1
            else:
                open_sessions[current_pc][4] = uw.loc[i]['time_stamp']
                open_sessions[current_pc][6].append(i)
        else:
            start_status = 1 if uw.loc[i]['act'] == 1 else 2
            open_sessions[current_pc] = [current_pc, start_status, 0, uw.loc[i]['time_stamp'], uw.loc[i]['time_stamp'], 1, [i]]
            if len(open_sessions) > 1: #increase the concurent count for all sessions
                for k in open_sessions:
                    open_sessions[k][5] +=1
    return sessions
                
def get_u_features_dicts(ul, data = 'r5.2'):
    ufdict = {}
    list_uf=[] if data in ['r4.1','r4.2'] else ['project']
    list_uf += ['role','b_unit','f_unit', 'dept','team']
    for f in list_uf:
        ul[f] = ul[f].astype(str)
        tmp = list(set(ul[f]))
        tmp.sort()
        ufdict[f] = {idx:i for i, idx in enumerate(tmp)}
    return (ul,ufdict, list_uf)

def proc_u_features(uf, ufdict, list_f = None, data = 'r4.2'): #to remove mode
    if type(list_f) != list:
        list_f=[] if data in ['r4.1','r4.2'] else ['project']
        list_f = ['role','b_unit','f_unit', 'dept','team'] + list_f

    out = []
    for f in list_f:
        out.append(ufdict[f][uf[f]])
    return out

def f_stats_calc(ud, fn, stats_f, countonly_f = {}, get_stats = False):
    f_count = len(ud)
    r = []
    f_names = []
    
    for f in stats_f:
        inp = ud[f].values
        if get_stats:
            if f_count > 0:
                r += [np.min(inp), np.max(inp), np.median(inp), np.mean(inp), np.std(inp)]
            else: r += [0, 0, 0, 0, 0]
            f_names += [fn+'_min_'+f, fn+'_max_'+f, fn+'_med_'+f, fn+'_mean_'+f, fn+'_std_'+f]
        else:
            if f_count > 0: r += [np.mean(inp)]
            else: r += [0]
            f_names += [fn+'_mean_'+f]
        
    for f in countonly_f:
        for v in countonly_f[f]:
            r += [sum(ud[f].values == v)]
            f_names += [fn+'_n-'+f+str(v)]
    return (f_count, r, f_names)

def f_calc_subfeatures(ud, fname, filter_col, filter_vals, filter_names, sub_features, countonly_subfeatures):
    [n, stats, fnames] = f_stats_calc(ud, fname,sub_features, countonly_subfeatures)
    allf = [n] + stats
    allf_names = ['n_'+fname] + fnames
    for i in range(len(filter_vals)):
        [n_sf, sf_stats, sf_fnames] = f_stats_calc(ud[ud[filter_col]==filter_vals[i]], filter_names[i], sub_features, countonly_subfeatures)
        allf += [n_sf] + sf_stats
        allf_names += [fname+'_n_'+filter_names[i]] + [fname + '_' + x for x in sf_fnames]
    return (allf, allf_names)

def f_calc(ud, mode = 'week', data = 'r4.2'):
    n_weekendact = (ud['time']==3).sum()
    if n_weekendact > 0: 
        is_weekend = 1
    else: 
        is_weekend = 0
    
    all_countonlyf = {'pc':[0,1,2,3]} if mode != 'session' else {}
    [all_f, all_f_names] = f_calc_subfeatures(ud, 'allact', None, [], [], [], all_countonlyf)
    if mode == 'day':
        [workhourf, workhourf_names] = f_calc_subfeatures(ud[(ud['time'] == 1) | (ud['time'] == 3)], 'workhourallact', None, [], [], [], all_countonlyf)
        [afterhourf, afterhourf_names] = f_calc_subfeatures(ud[(ud['time'] == 2) | (ud['time'] == 4) ], 'afterhourallact', None, [], [], [], all_countonlyf)
    elif mode == 'week':
        [workhourf, workhourf_names] = f_calc_subfeatures(ud[ud['time'] == 1], 'workhourallact', None, [], [], [], all_countonlyf)
        [afterhourf, afterhourf_names] = f_calc_subfeatures(ud[ud['time'] == 2 ], 'afterhourallact', None, [], [], [], all_countonlyf)
        [weekendf, weekendf_names] = f_calc_subfeatures(ud[ud['time'] >= 3 ], 'weekendallact', None, [], [], [], all_countonlyf)

    logon_countonlyf = {'pc':[0,1,2,3]} if mode != 'session' else {}
    logon_statf = []
        
    [all_logonf, all_logonf_names] = f_calc_subfeatures(ud[ud['act']==1], 'logon', None, [], [], logon_statf, logon_countonlyf)
    if mode == 'day':
        [workhourlogonf, workhourlogonf_names] = f_calc_subfeatures(ud[(ud['act']==1) & ((ud['time'] == 1) | (ud['time'] == 3) )], 'workhourlogon', None, [], [], logon_statf, logon_countonlyf)
        [afterhourlogonf, afterhourlogonf_names] = f_calc_subfeatures(ud[(ud['act']==1) & ((ud['time'] == 2) | (ud['time'] == 4) )], 'afterhourlogon', None, [], [], logon_statf, logon_countonlyf)
    elif mode == 'week':
        [workhourlogonf, workhourlogonf_names] = f_calc_subfeatures(ud[(ud['act']==1) & (ud['time'] == 1)], 'workhourlogon', None, [], [], logon_statf, logon_countonlyf)
        [afterhourlogonf, afterhourlogonf_names] = f_calc_subfeatures(ud[(ud['act']==1) & (ud['time'] == 2) ], 'afterhourlogon', None, [], [], logon_statf, logon_countonlyf)
        [weekendlogonf, weekendlogonf_names] = f_calc_subfeatures(ud[(ud['act']==1) & (ud['time'] >= 3) ], 'weekendlogon', None, [], [], logon_statf, logon_countonlyf)
    
    device_countonlyf = {'pc':[0,1,2,3]} if mode != 'session' else {}
    device_statf = ['usb_dur','file_tree_len'] if data not in ['r4.1','r4.2'] else ['usb_dur']
        
    [all_devicef, all_devicef_names] = f_calc_subfeatures(ud[ud['act']==3], 'usb', None, [], [], device_statf, device_countonlyf)
    if mode == 'day':
        [workhourdevicef, workhourdevicef_names] = f_calc_subfeatures(ud[(ud['act']==3) & ((ud['time'] == 1) | (ud['time'] == 3) )], 'workhourusb', None, [], [], device_statf, device_countonlyf)
        [afterhourdevicef, afterhourdevicef_names] = f_calc_subfeatures(ud[(ud['act']==3) & ((ud['time'] == 2) | (ud['time'] == 4) )], 'afterhourusb', None, [], [], device_statf, device_countonlyf)
    elif mode == 'week':
        [workhourdevicef, workhourdevicef_names] = f_calc_subfeatures(ud[(ud['act']==3) & (ud['time'] == 1)], 'workhourusb', None, [], [], device_statf, device_countonlyf)
        [afterhourdevicef, afterhourdevicef_names] = f_calc_subfeatures(ud[(ud['act']==3) & (ud['time'] == 2) ], 'afterhourusb', None, [], [], device_statf, device_countonlyf)
        [weekenddevicef, weekenddevicef_names] = f_calc_subfeatures(ud[(ud['act']==3) & (ud['time'] >= 3) ], 'weekendusb', None, [], [], device_statf, device_countonlyf)
          
    if mode != 'session': file_countonlyf = {'to_usb':[1],'from_usb':[1], 'file_act':[1,2,3,4], 'disk':[0,1], 'pc':[0,1,2,3]}
    else: file_countonlyf = {'to_usb':[1],'from_usb':[1], 'file_act':[1,2,3,4], 'disk':[0,1,2]}
    if data in ['r4.1','r4.2']: 
        [file_countonlyf.pop(k) for k in ['to_usb','from_usb', 'file_act']]
    
    (all_filef, all_filef_names) = f_calc_subfeatures(ud[ud['act']==7], 'file', 'file_type', [1,2,3,4,5,6], \
            ['otherf','compf','phof','docf','txtf','exef'], ['file_len', 'file_depth', 'file_nwords'], file_countonlyf)
    
    if mode == 'day':
        (workhourfilef, workhourfilef_names) = f_calc_subfeatures(ud[(ud['act']==7) & ((ud['time'] ==1) | (ud['time'] ==3))], 'workhourfile', 'file_type', [1,2,3,4,5,6], ['otherf','compf','phof','docf','txtf','exef'], ['file_len', 'file_depth', 'file_nwords'], file_countonlyf)
        (afterhourfilef, afterhourfilef_names) = f_calc_subfeatures(ud[(ud['act']==7) & ((ud['time'] ==2) | (ud['time'] ==4))], 'afterhourfile', 'file_type', [1,2,3,4,5,6], ['otherf','compf','phof','docf','txtf','exef'], ['file_len', 'file_depth', 'file_nwords'], file_countonlyf)
    elif mode == 'week':
        (workhourfilef, workhourfilef_names) = f_calc_subfeatures(ud[(ud['act']==7) & (ud['time'] ==1)], 'workhourfile', 'file_type', [1,2,3,4,5,6], ['otherf','compf','phof','docf','txtf','exef'], ['file_len', 'file_depth', 'file_nwords'], file_countonlyf)
        (afterhourfilef, afterhourfilef_names) = f_calc_subfeatures(ud[(ud['act']==7) & (ud['time'] ==2)], 'afterhourfile', 'file_type', [1,2,3,4,5,6], ['otherf','compf','phof','docf','txtf','exef'], ['file_len', 'file_depth', 'file_nwords'], file_countonlyf)
        (weekendfilef, weekendfilef_names) = f_calc_subfeatures(ud[(ud['act']==7) & (ud['time'] >= 3)], 'weekendfile', 'file_type', [1,2,3,4,5,6], ['otherf','compf','phof','docf','txtf','exef'], ['file_len', 'file_depth', 'file_nwords'], file_countonlyf)
        
    email_stats_f = ['n_des', 'n_atts', 'n_exdes', 'n_bccdes', 'email_size', 'email_text_slen', 'email_text_nwords']
    if data not in ['r4.1','r4.2']:
        email_stats_f += ['e_att_other', 'e_att_comp', 'e_att_pho', 'e_att_doc', 'e_att_txt', 'e_att_exe']
        email_stats_f += ['e_att_sother', 'e_att_scomp', 'e_att_spho', 'e_att_sdoc', 'e_att_stxt', 'e_att_sexe'] 
        mail_filter = 'send_mail'
        mail_filter_vals = [0,1]
        mail_filter_names = ['recvmail','send_mail']
    else:
        mail_filter, mail_filter_vals, mail_filter_names = None, [], []    
    
    if mode != 'session': mail_countonlyf = {'Xemail':[1],'exbccmail':[1], 'pc':[0,1,2,3]}
    else: mail_countonlyf = {'Xemail':[1],'exbccmail':[1]}
    
    (all_emailf, all_emailf_names) = f_calc_subfeatures(ud[ud['act']==6], 'email', mail_filter, mail_filter_vals, mail_filter_names , email_stats_f, mail_countonlyf)
    if mode == 'week':
        (workhouremailf, workhouremailf_names) = f_calc_subfeatures(ud[(ud['act']==6) & (ud['time'] == 1)], 'workhouremail', mail_filter, mail_filter_vals, mail_filter_names, email_stats_f, mail_countonlyf)
        (afterhouremailf, afterhouremailf_names) = f_calc_subfeatures(ud[(ud['act']==6) & (ud['time'] == 2)], 'afterhouremail', mail_filter, mail_filter_vals, mail_filter_names, email_stats_f, mail_countonlyf)
        (weekendemailf, weekendemailf_names) = f_calc_subfeatures(ud[(ud['act']==6) & (ud['time'] >= 3)], 'weekendemail', mail_filter, mail_filter_vals, mail_filter_names, email_stats_f, mail_countonlyf)
    elif mode == 'day':
        (workhouremailf, workhouremailf_names) = f_calc_subfeatures(ud[(ud['act']==6) & ((ud['time'] ==1) | (ud['time'] ==3))], 'workhouremail', mail_filter, mail_filter_vals, mail_filter_names, email_stats_f, mail_countonlyf)
        (afterhouremailf, afterhouremailf_names) = f_calc_subfeatures(ud[(ud['act']==6) & ((ud['time'] ==2) | (ud['time'] ==4))], 'afterhouremail', mail_filter, mail_filter_vals, mail_filter_names, email_stats_f, mail_countonlyf)    
    
    if data in ['r5.2','r5.1'] or data in ['r4.1','r4.2']:
        http_count_subf =  {'pc':[0,1,2,3]}
    elif data in ['r6.2','r6.1']:
        http_count_subf = {'pc':[0,1,2,3], 'http_act':[1,2,3]}
    
    if mode == 'session': http_count_subf.pop('pc',None)

    (all_httpf, all_httpf_names) = f_calc_subfeatures(ud[ud['act']==5], 'http', 'http_type', [1,2,3,4,5,6], \
            ['otherf','socnetf','cloudf','jobf','leakf','hackf'], ['url_len', 'url_depth', 'http_c_len', 'http_c_nwords'], http_count_subf)
    
    if mode == 'week':
        (workhourhttpf, workhourhttpf_names) = f_calc_subfeatures(ud[(ud['act']==5) & (ud['time'] ==1)], 'workhourhttp', 'http_type', [1,2,3,4,5,6], \
                ['otherf','socnetf','cloudf','jobf','leakf','hackf'], ['url_len', 'url_depth', 'http_c_len', 'http_c_nwords'], http_count_subf)
        (afterhourhttpf, afterhourhttpf_names) = f_calc_subfeatures(ud[(ud['act']==5) & (ud['time'] ==2)], 'afterhourhttp', 'http_type', [1,2,3,4,5,6], \
                ['otherf','socnetf','cloudf','jobf','leakf','hackf'], ['url_len', 'url_depth', 'http_c_len', 'http_c_nwords'], http_count_subf)
        (weekendhttpf, weekendhttpf_names) = f_calc_subfeatures(ud[(ud['act']==5) & (ud['time'] >=3)], 'weekendhttp', 'http_type', [1,2,3,4,5,6], \
                ['otherf','socnetf','cloudf','jobf','leakf','hackf'], ['url_len', 'url_depth', 'http_c_len', 'http_c_nwords'], http_count_subf)
    elif mode == 'day':
        (workhourhttpf, workhourhttpf_names) = f_calc_subfeatures(ud[(ud['act']==5) & ((ud['time'] ==1) | (ud['time'] ==3))], 'workhourhttp', 'http_type', [1,2,3,4,5,6], \
                ['otherf','socnetf','cloudf','jobf','leakf','hackf'], ['url_len', 'url_depth', 'http_c_len', 'http_c_nwords'], http_count_subf)
        (afterhourhttpf, afterhourhttpf_names) = f_calc_subfeatures(ud[(ud['act']==5) & ((ud['time'] ==2) | (ud['time'] ==4))], 'afterhourhttp', 'http_type', [1,2,3,4,5,6], \
                ['otherf','socnetf','cloudf','jobf','leakf','hackf'], ['url_len', 'url_depth', 'http_c_len', 'http_c_nwords'], http_count_subf)
        
    numActs = all_f[0]
    mal_u = 0
    if (ud['mal_act']).sum() > 0:
        tmp = list(set(ud['insider']))
        if len(tmp) > 1:
            tmp.remove(0.0)
        mal_u = tmp[0]
        
    if mode == 'week':        
        features_tmp =  all_f + workhourf + afterhourf + weekendf +\
                        all_logonf + workhourlogonf + afterhourlogonf + weekendlogonf +\
                        all_devicef + workhourdevicef + afterhourdevicef + weekenddevicef +\
                        all_filef + workhourfilef + afterhourfilef + weekendfilef + \
                        all_emailf + workhouremailf + afterhouremailf + weekendemailf + all_httpf + workhourhttpf + afterhourhttpf + weekendhttpf
        fnames_tmp = all_f_names + workhourf_names + afterhourf_names + weekendf_names +\
                      all_logonf_names + workhourlogonf_names + afterhourlogonf_names + weekendlogonf_names +\
                      all_devicef_names + workhourdevicef_names + afterhourdevicef_names + weekenddevicef_names +\
                      all_filef_names + workhourfilef_names + afterhourfilef_names + weekendfilef_names + \
                      all_emailf_names + workhouremailf_names + afterhouremailf_names + weekendemailf_names + all_httpf_names + workhourhttpf_names + afterhourhttpf_names + weekendhttpf_names
    elif mode == 'day':
        features_tmp = all_f + workhourf + afterhourf +\
                        all_logonf + workhourlogonf + afterhourlogonf +\
                        all_devicef + workhourdevicef + afterhourdevicef + \
                        all_filef + workhourfilef + afterhourfilef + \
                        all_emailf + workhouremailf + afterhouremailf + all_httpf + workhourhttpf + afterhourhttpf
        fnames_tmp = all_f_names + workhourf_names + afterhourf_names +\
                      all_logonf_names + workhourlogonf_names + afterhourlogonf_names +\
                      all_devicef_names + workhourdevicef_names + afterhourdevicef_names +\
                      all_filef_names + workhourfilef_names + afterhourfilef_names + \
                      all_emailf_names + workhouremailf_names + afterhouremailf_names + all_httpf_names + workhourhttpf_names + afterhourhttpf_names
    elif mode == 'session':
        features_tmp = all_f + all_logonf + all_devicef + all_filef + all_emailf + all_httpf
        fnames_tmp = all_f_names + all_logonf_names + all_devicef_names + all_filef_names + all_emailf_names + all_httpf_names
    
    return [numActs, is_weekend, features_tmp, fnames_tmp, mal_u]

def session_instance_calc(ud, sinfo, week, mode, data, uw, v, list_uf):
    d = ud.iloc[0]['day']
    perworkhour = sum(ud['time']==1)/len(ud)
    perafterhour = sum(ud['time']==2)/len(ud)
    perweekend = sum(ud['time']==3)/len(ud)
    perweekendafterhour = sum(ud['time']==4)/len(ud)
    st_timestamp = min(ud['time_stamp'])
    end_timestamp = max(ud['time_stamp'])
    s_dur = (end_timestamp - st_timestamp).total_seconds() / 60 # in minute
    s_start = st_timestamp.hour + st_timestamp.minute/60
    s_end = end_timestamp.hour + end_timestamp.minute/60
    starttime = st_timestamp.timestamp()
    endtime = end_timestamp.timestamp()
    n_days = len(set(ud['day']))        
    
    tmp = f_calc(ud, mode, data)
    session_instance = [starttime, endtime, v, sinfo[0], d, week, ud.iloc[0]['pc'], perworkhour, perafterhour, perweekend,
                        perweekendafterhour, n_days, s_dur, sinfo[6], sinfo[2], sinfo[3], s_start, s_end] + \
        (uw.loc[v, list_uf + ['ITAdmin', 'O', 'C', 'E', 'A', 'N'] ]).tolist() + tmp[2] + [tmp[4]]
    return (session_instance, tmp[3])

def to_csv(week, mode, data, ul, uf_dict, list_uf, subsession_mode = {}):
    user_dict = {i : idx for (i, idx) in enumerate(ul.index)} 
    if mode == 'session': 
        first_sid = week*100000 # to get an unique index for each session, also, first 1 or 2 number in index would be week number
        cols2a = ['starttime', 'endtime','user', 'sessionid', 'day', 'week', 'pc', 'isworkhour', 'isafterhour','isweekend', 
                  'isweekendafterhour', 'n_days', 'duration', 'n_concurrent_sessions', 'start_with', 'end_with', 'ses_start', 
                  'ses_end'] + list_uf + ['ITAdmin','O','C','E','A','N']
    elif mode == 'day': 
        cols2a = ['starttime', 'endtime','user', 'day', 'week', 'isweekday','isweekend'] + list_uf +\
            ['ITAdmin','O','C','E','A','N']
    else: cols2a = ['starttime', 'endtime','user','week'] + list_uf + ['ITAdmin','O','C','E','A','N']
    cols2b = ['insider']        

    w = pd.read_pickle("NumDataByWeek/"+str(week)+"_num.pickle")

    usnlist = list(set(w['user'].astype('int').values))
    if True:
        cols = ['week']+ list_uf + ['ITAdmin', 'O', 'C', 'E', 'A', 'N', 'insider'] 
        uw = pd.DataFrame(columns = cols, index = user_dict.keys())
        uwdict = {}
        for v in user_dict:
            if v in usnlist:
                is_ITAdmin = 1 if ul.loc[user_dict[v], 'role'] == 'ITAdmin' else 0
                row = [week] + proc_u_features(ul.loc[user_dict[v]], uf_dict, list_uf, data = data) + [is_ITAdmin] + \
                    (ul.loc[user_dict[v],['O','C','E','A','N']]).tolist() + [0]
                row[-1] = int(list(set(w[w['user']==v]['insider']))[0])
                uwdict[v] = row
        uw = pd.DataFrame.from_dict(uwdict, orient = 'index',columns = cols)    
    
    towrite = pd.DataFrame()
    towrite_list = []
    
    if mode == 'session' and len(subsession_mode) > 0:
        towrite_list_subsession = {} 
        for k1 in subsession_mode:
            towrite_list_subsession[k1] = {}
            for k2 in subsession_mode[k1]:
                towrite_list_subsession[k1][k2] = []
    
    days = list(set(w['day']))
    for v in user_dict:
        if v in usnlist:
            uactw = w[w['user']==v]
            
            if mode == 'week':
                a = uactw.iloc[0]['time_stamp']
                a = a - timedelta(int(a.strftime("%w"))) # get the nearest Sunday
                starttime = datetime(a.year, a.month, a.day).timestamp()
                endtime = (datetime(a.year, a.month, a.day) + timedelta(days=7)).timestamp()
                
                if len(uactw) > 0:
                    tmp = f_calc(uactw, mode, data)
                    i_fnames = tmp[3]
                    towrite_list.append([starttime, endtime, v, week] + (uw.loc[v, list_uf + ['ITAdmin', 'O', 'C', 'E', 'A', 'N'] ]).tolist() + tmp[2] + [ tmp[4]])

            if mode == 'session':
                sessions = get_sessions(uactw, first_sid)
                first_sid += len(sessions)
                for s in sessions:
                    sinfo = sessions[s]
                    
                    ud = uactw.loc[sessions[s][7]]
                    if len(ud) > 0:                     
                        session_instance, i_fnames = session_instance_calc(ud, sinfo, week, mode, data, uw, v, list_uf)
                        towrite_list.append(session_instance)
                        
                        ## do subsessions:
                        if 'time' in subsession_mode: # divide a session into subsessions by consecutive time chunks
                            for subsession_dur in subsession_mode['time']:
                                n_subsession = int(np.ceil(session_instance[12] / subsession_dur))
                                if n_subsession == 1:
                                    towrite_list_subsession['time'][subsession_dur].append([0] + session_instance)
                                else:
                                    sinfo1 = sinfo.copy()
                                    for subsession_ind in range(n_subsession):
                                        sinfo1[3] = 0 if subsession_ind < n_subsession-1 else sinfo[3] 
                                        
                                        subsession_ud = ud[(ud['time_stamp'] >= sessions[s][4] + timedelta(minutes = subsession_ind*subsession_dur)) & \
                                                            (ud['time_stamp'] < sessions[s][4] + timedelta(minutes = (subsession_ind+1)*subsession_dur))]
                                        if len(subsession_ud) > 0:
                                            ss_instance, _ = session_instance_calc(subsession_ud, sinfo1, week, mode, data, uw, v, list_uf)
                                            towrite_list_subsession['time'][subsession_dur].append([subsession_ind] + ss_instance)
                            
                        if 'nact' in subsession_mode:
                            for ss_nact in subsession_mode['nact']:
                                n_subsession = int(np.ceil(len(ud) / ss_nact))
                                if n_subsession == 1:
                                    towrite_list_subsession['nact'][ss_nact].append([0] + session_instance)
                                else:
                                    sinfo1 = sinfo.copy()
                                    for ss_ind in range(n_subsession):
                                        sinfo1[3] = 0 if ss_ind < n_subsession-1 else sinfo[3] 
                                        
                                        ss_ud = ud.iloc[ss_ind*ss_nact : min(len(ud), (ss_ind+1)*ss_nact)] 
                                        if len(ss_ud) > 0:
                                            ss_instance,_ = session_instance_calc(ss_ud, sinfo1, week, mode, data, uw, v, list_uf)
                                            towrite_list_subsession['nact'][ss_nact].append([ss_ind] + ss_instance)
                        
            if mode == 'day':
                days = sorted(list(set(uactw['day']))) 
                for d in days:
                    ud = uactw[uactw['day'] == d]
                    isweekday = 1 if sum(ud['time']>=3) == 0 else 0
                    isweekend = 1-isweekday
                    a = ud.iloc[0]['time_stamp']
                    starttime = datetime(a.year, a.month, a.day).timestamp()
                    endtime = (datetime(a.year, a.month, a.day) + timedelta(days=1)).timestamp()
                    
                    if len(ud) > 0:
                        tmp = f_calc(ud, mode, data)
                        i_fnames = tmp[3]
                        towrite_list.append([starttime, endtime, v, d, week, isweekday, isweekend] + (uw.loc[v, list_uf + ['ITAdmin', 'O', 'C', 'E', 'A', 'N'] ]).tolist() + tmp[2] + [ tmp[4]])

    towrite = pd.DataFrame(columns = cols2a + i_fnames + cols2b, data = towrite_list)
    towrite.to_pickle("tmp/"+str(week) + mode+".pickle")
    
    if mode == 'session' and len(subsession_mode) > 0:
        for k1 in subsession_mode:
            for k2 in subsession_mode[k1]:
                df_tmp = pd.DataFrame(columns = ['subs_ind']+cols2a + i_fnames + cols2b, data = towrite_list_subsession[k1][k2])
                df_tmp.to_pickle("tmp/"+str(week) + mode + k1 + str(k2) + ".pickle")
    
if __name__ == "__main__":
    dname = os.getcwd().split('/')[-1]
    if dname not in ['r4.1','r4.2','r6.2','r6.1','r5.1','r5.2']:
        raise Exception('Please put this script in and run it from a CERT data folder (e.g. r4.2)')
    #make temporary folders
    [os.mkdir(x) for x in ["tmp", "ExtractedData", "DataByWeek", "NumDataByWeek"]]
    
    subsession_mode = {'nact':[25, 50], 'time':[120, 240]}#this can be an empty dict
    
    numCores = 8
    arguments = len(sys.argv) - 1
    if arguments > 0:
        numCores = int(sys.argv[1])
        
    numWeek = 73 if dname in ['r4.1','r4.2'] else 75 # only 73 weeks in r4.1 and r4.2 dataset
    st = time.time()
    
    #### Step 1: Combine data from sources by week, stored in DataByWeek
    combine_by_timerange_pandas(dname)
    print(f"Step 1 - Separate data by week - done. Time (mins): {(time.time()-st)/60:.2f}")
    st = time.time()
    
    #### Step 2: Get user list
    users = get_mal_userdata(dname)
    print(f"Step 2 - Get user list - done. Time (mins): {(time.time()-st)/60:.2f}")
    st = time.time()
    
    #### Step 3: Convert each action to numerical data, stored in NumDataByWeek
    Parallel(n_jobs=numCores)(delayed(process_week_num)(i, users, data=dname) for i in range(numWeek))
    print(f"Step 3 - Convert each action to numerical data - done. Time (mins): {(time.time()-st)/60:.2f}")
    st = time.time()
    
    #### Step 4: Extract to csv
    for mode in ['week','day','session']:
    
        weekRange = list(range(0, numWeek)) if mode in ['day', 'session'] else list(range(1, numWeek))
        (ul, uf_dict, list_uf) = get_u_features_dicts(users, data= dname)
        
        Parallel(n_jobs=numCores)(delayed(to_csv)(i, mode, dname, ul, uf_dict, list_uf, subsession_mode) 
                                   for i in weekRange)

        all_csv = open('ExtractedData/'+mode+dname+'.csv','a')
        
        towrite = pd.read_pickle("tmp/"+str(weekRange[0]) + mode+".pickle")
        towrite.to_csv(all_csv,header=True, index = False)
        for w in weekRange[1:]:
            towrite = pd.read_pickle("tmp/"+str(w) + mode+".pickle")        
            towrite.to_csv(all_csv,header=False, index = False)
        
        if mode == 'session' and len(subsession_mode) > 0:
            for k1 in subsession_mode:
                for k2 in subsession_mode[k1]:
                    all_csv = open('ExtractedData/'+mode+ k1 + str(k2) + dname+'.csv','a')
                    towrite = pd.read_pickle('tmp/'+str(weekRange[0]) + mode + k1 + str(k2)+".pickle")
                    towrite.to_csv(all_csv,header=True, index = False)
                    for w in weekRange[1:]:
                        towrite = pd.read_pickle('tmp/'+str(w) + mode+ k1 + str(k2)+".pickle")        
                        towrite.to_csv(all_csv,header=False, index = False)
                    
        print(f'Extracted {mode} data. Time (mins): {(time.time()-st)/60:.2f}')
        st = time.time()

    [os.system(f"rm -r {x}") for x in ["tmp", "DataByWeek", "NumDataByWeek"]]
