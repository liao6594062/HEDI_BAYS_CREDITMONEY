# -*- coding: utf-8 -*-
"""
Created on Mon May 27 10:39:05 2019

@author: wj61032
"""
import csv
import pandas as pd
import numpy as np
import hyperopt.pyll.stochastic 
from timeit import default_timer as timer
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

#数据地址
data_path=r'\\file.tounawang.net\风控数据共享\llll\D_HE\3_贝叶斯优化出额\水畔按揭车出额\data'+'\\'

#读取数据
opt_data=pd.read_excel(data_path+'水畔_按揭车出额.xlsx')
opt_data['分数分段'].replace({'(0.0, 8.828]':'A档_(0.0, 8.828]',
                            '(8.828, 11.329]':'B档_(8.828, 11.329]',
                            '(11.329, 15.762]':'C档_(11.329, 15.762]',
                            '(15.762, 100.0]':'D档_(15.762, 100.0]'},
                            inplace=True)
cline_show = opt_data[['车三百评估价','loanamt']].groupby(opt_data['分数分段']).mean().sort_index()

'''
最优化
===============================================================================
'''
#评分分四段，每段设置参数空间
D = np.arange(0.11,0.41,0.01)
C = np.arange(0.11,0.41,0.01)
B = np.arange(0.11,0.41,0.01)
A = np.arange(0.11,0.51,0.01)
'''构造参数空间'''
space = {'D档_(15.762, 100.0]': hp.choice('D档_(15.762, 100.0]',D), 
    'C档_(11.329, 15.762]': hp.choice('C档_(11.329, 15.762]',C),
    'B档_(8.828, 11.329]': hp.choice('B档_(8.828, 11.329]',B),
    'A档_(0.0, 8.828]':hp.choice('A档_(0.0, 8.828]',A)}   

#查看参数空间中一个点
exp_ = hyperopt.pyll.stochastic.sample(space)
#参看搜索空间总数
len_space = len(D)*len(C)*len(B)*len(A)
print(len_space)

def objective(hyperparameters):
    """创建最优化的目标函数"""
    
    global ITERATION    
    ITERATION += 1
    
    start = timer() 
    #计算各档基础额度
    opt_data['基础成数'] = opt_data['分数分段'].replace(hyperparameters)
    opt_data['额度'] = opt_data['车三百评估价']*opt_data['基础成数']  
    #添加边界条件 
    def boundary(x):
        '''超过边界条件设置为边界条件'''
        if x>=80000:
            res_ =80000
        elif x<=30000:
            res_=30000
        else:
            res_=x
        return res_
    opt_data['最终额度'] = opt_data['额度'].apply(boundary)    
    opt_data['逾期金额'] = np.where(opt_data['当前逾期15+']==1,
            opt_data['最终额度'],0)
    
    res_ = opt_data['最终额度'].groupby(opt_data['分数分段']).mean().to_dict()

    run_time = timer() - start
    
    # 获取得分及std
    fpd4_num = opt_data['当前逾期15+'].mean()
    fpd4_loan = opt_data['逾期金额'].sum()/opt_data['最终额度'].sum()
    cline_mean = opt_data['最终额度'].mean()
    #最优化目标
    loss = fpd4_loan/fpd4_num    

    # 结果写入本地
    of_connection = open(OUT_FILE, 'a')
    writer = csv.writer(of_connection)
    writer.writerow([loss, hyperparameters, ITERATION, run_time, 
                     fpd4_num,fpd4_loan,cline_mean,res_])
    of_connection.close()
    
    return {'loss': loss, 
            'hyperparameters': hyperparameters, 
            'iteration': ITERATION,
            'train_time': run_time, 
            'status': STATUS_OK,
            'fpd4_loan':fpd4_loan,
            'cline_mean':cline_mean}
    

# 存储信息的构造
trials = Trials() 
#过程存储
path=r'E:\水畔按揭车201907'+'\\'
OUT_FILE = path+'按揭车额度_贝叶斯优化.csv'        
of_connection = open(OUT_FILE,'w')
writer = csv.writer(of_connection)
ITERATION=0
headers = ['loss', 'hyperparameters', 'iteration', 'runtime', 
           '笔数_当前逾期15+', '金额_当前逾期15+','cline_mean','res_']
writer.writerow(headers)
of_connection.close()

# 构造fmin
best = fmin(
    fn=objective,
    space=space,
    algo=tpe.suggest,
    max_evals=100000,
    trials=trials)
print(best)

data = pd.read_csv(data_path+'按揭车额度_贝叶斯优化.csv',engine='python')
data_out = data[(data['cline_mean']<=42000)&(data['cline_mean']>38000)].sort_values(['金额_当前与逾期15+'])
data_out.to_excel(data_path+'额度最优解.xlsx')
