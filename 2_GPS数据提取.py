# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 15:22:14 2018

@author: wj61032
"""

import numpy as np
import pandas as pd
from datetime import datetime

#数据地址
path = r'\\file.tounawang.net\风控数据共享\llll\D_HE\1_车贷委外名单筛选\data'+'\\'

'''进度逾期表数据读取，数据时间需要每月修改'''
data1 = pd.read_csv(path+'Tbdsche_over20190829.csv',
                    engine='python') # 进度逾期表
data1.columns = [x.lower() for x in data1.columns]

data=data1
# 规则0参数：当前逾期天数>15
low_days = 15
# 规则1参数：产品类型
loantypes = ['车贷','车速贷','车信易贷',
             '飞车贷','过户车贷','秒贷',
             '闪贷','随借随还-车贷']
# 根据规则筛选
data_0 = data[data['currentduedays']>low_days] #逾期天数>15
data_1 = data_0[data_0['loantype'].isin(loantypes)] #产品类型

'''委外解密数据读取，数据时间需要每月修改'''
data_out = pd.read_excel(path+r'委外数据解密_20190830.xlsx') 

# 筛选后的进度逾期表和委外数据合并
df = pd.merge(data_1,data_out,left_on='contractno',right_on='合同号',how='left')
#需要提取GPS的数据保存
df[['contractno','车牌号']].to_excel(path+'201909GPS信息提取.xlsx')












