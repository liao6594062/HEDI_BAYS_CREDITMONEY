# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 15:46:10 2019

@author: wj61032
"""
import sys
sys.path.append(r'\\file.tounawang.net\风控数据共享\小工具\internal_lib\treecut\scripts')
import pandas as pd
import numpy as np
from tree_cut import *
from datetime import datetime

data_path = r'\\file.tounawang.net\风控数据共享\llll\D_HE\4_三方数据测试\黑瞳测试\data'+'\\'

'''数据载入'''
df = pd.read_excel(data_path+'黑瞳测试数据.xlsx')

'''设置参考用变量'''
np.random.seed(666)
df['随机数_参考用'] = np.random.rand(len(df))

cols = [x for x in df if x not in ['最大逾期7+','放款年月']]

#变量测试
ivs = woe_show(df,cols,data_path,
               file_name='黑瞳_单变量测试_最大逾期7+.xlsx',
               label='放款年月',
               max_groups=6,target='最大逾期7+')
'''
df：包含X和y的数据集；
target：目标变量名;
path：结果写入地址；
col_list：需要查看的变量列表，数值型变量；
max_groups：最大分箱数;
label：分月查看字段名，默认为放款年月
'''



