# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 11:27:30 2019

@author: wj61032
"""

import pandas as pd
import numpy as np
from impala.dbapi import connect
from impala.util import as_pandas
from datetime import datetime,timedelta


def qurry_hdp(sql,flag=1):
    '''select * 时传入flag=0,其他时候无需传入flag'''
    conn = connect(host='172.30.17.197', port=10000, 
               user='hedi', auth_mechanism='PLAIN', 
               password='9ku3y3b9')
    cur = conn.cursor()
    cur.execute(sql)
    temp=as_pandas(cur)
    cur.close()
    conn.close()
    if flag != 1:
        temp.columns = [x.split('.')[1] for x in temp.columns]
    return temp

sql ="""
select * from provider_api_db.tio_decision_credit_auto_merge
where data_date>='2019-08-03'
and ck_app_node in ('CKD_BASIC_CERT','CKD_MANUAL_CERT','CKD_ADVANCED_CERT')
"""

deci_data = qurry_hdp(sql,0)

col_dict = pd.read_excel(r'\\file.tounawang.net\分析中心\hedi\很好借\51公积金分析\51公积金分析20190726\data\dict_.xlsx')
col_dict = dict(zip(col_dict['col_name'],col_dict['label']))
deci_data.rename(columns=col_dict,inplace=True)

'''加工逻辑验证'''
ld = deci_data.copy()

def col0(x):    
    res = -0.3722
    if x <39.5 and x>0:
        res = 0.1905
    return res
col = '埋点类.OCR.年龄'
ld['woe_'+col] = ld[col].apply(col0)

ld['百融.多头.按身份证号查询，近3个月在银行机构最大月申请次数']=ld['百融.多头.按身份证号查询，近3个月在银行机构最大月申请次数'].replace({'':np.nan}).astype(float)
def col1(x):    
    res = 0.2205
    if x>1.5:
        res = -0.4209
    return res
col='百融.多头.按身份证号查询，近3个月在银行机构最大月申请次数'
ld['woe_'+col] = ld[col].apply(col1)

def col2(x):    
    res = -1.0149
    if x>662.5:
        res = 1.5614
    elif x>484.5:
        res = 0.1490
    return res
col='同盾智信分.小金分.小额现金贷分'
ld['woe_'+col] = ld[col].apply(col2)

def col3(x):    
    res = -0.5760
    if x>43.235:
        res = 0.1940
    return res
col='聚信利.手机核查信息.近6月话费消费平均值'
ld['woe_'+col] = ld[col].apply(col3)

def col4(x):    
    res = -0.2263
    if x>717.5:
        res = 0.6966
    elif x>701.5:
        res=0.4484
    return res
col='芝麻分'
ld[col] = ld[col].replace('',np.nan).astype(float)
ld['woe_'+col] = ld[col].apply(col4)

def col5(x):    
    res = -0.2304
    if x>0.5:
        res = 0.6781
    elif x>=0:
        res=-0.0879
    return res
col='支付结算协会.收到借款申请机构总数（6个月）'
ld['woe_'+col] = ld[col].apply(col5)

def col6(x):    
    res = -0.4117
    if x>4.5:
        res = 1.6632
    elif x>=0:
        res = 0.1276
    return res
col='新颜.负面拉黑.睡眠机构数'
ld['woe_'+col] = ld[col].apply(col6)

def col7(x):    
    res = 0.0860
    if x>0.2:
        res = -0.7177
    return res
col='宜信.违约概率'
ld['woe_'+col] = ld[col].apply(col7)

def col8(x):    
    res = -0.2423
    if x>905.5:
        res = 0.3869
    return res
col='51公积金.分析得到的月缴额'
ld[col] = ld[col].astype(float)
ld['woe_'+col] = ld[col].apply(col8)

def col9(x):    
    res = -0.1037
    if x>622:
        res = 1.1318
    return res
col='畅快贷.通讯录.通讯录联系人总个数'
ld['woe_'+col] = ld[col].apply(col9)

def col10(x):    
    res = -0.2667
    if x>=0 and x<=5.5:
        res = 0.1940
    return res
col='蜜罐.180天内历史查询.现金贷查询机构数'
ld['woe_'+col] = ld[col].apply(col10)

def col11(x):    
    res = -0.2199
    if x>=0 and x<=4.5:
        res = 0.1823
    return res
col='宜信.查询时间为三个月内其他机构查询次数'
ld['woe_'+col] = ld[col].apply(col11)

def col12(x):    
    res = -0.103
    if x>64.5:
        res = 0.6667
    return res
col='蜜罐.主动联系的种等亲密联系人数'
ld['woe_'+col] = ld[col].apply(col12)

def col13(x):    
    res = -0.0786
    if x>33.5:
        res = 0.5428
    return res
col='蜜罐.主动联系的亲密联系人数'
ld['woe_'+col] = ld[col].apply(col13)

ld['z'] = -2.77189596\
            -1.1666815*ld['woe_埋点类.OCR.年龄']\
            -0.89955485*ld['woe_百融.多头.按身份证号查询，近3个月在银行机构最大月申请次数']\
            -0.87875999*ld['woe_同盾智信分.小金分.小额现金贷分']\
            -0.71693423*ld['woe_聚信利.手机核查信息.近6月话费消费平均值']\
            -0.66392102*ld['woe_芝麻分']\
            -0.620809*ld['woe_支付结算协会.收到借款申请机构总数（6个月）']\
            -0.59130557*ld['woe_新颜.负面拉黑.睡眠机构数']\
            -0.59119111*ld['woe_宜信.违约概率']\
            -0.55485498*ld['woe_51公积金.分析得到的月缴额']\
            -0.53735721*ld['woe_畅快贷.通讯录.通讯录联系人总个数']\
            -0.53338044*ld['woe_蜜罐.180天内历史查询.现金贷查询机构数']\
            -0.52795054*ld['woe_宜信.查询时间为三个月内其他机构查询次数']\
            -0.46267159*ld['woe_蜜罐.主动联系的种等亲密联系人数']\
            -0.32262149*ld['woe_蜜罐.主动联系的亲密联系人数']            
ld['m_score'] = ld['z'].apply(lambda x: 100/(1+np.exp(-x)))

debug = ld[['m_score','credit_51benchmark_score']]

debug = ld[['woe_埋点类.OCR.年龄',
        'woe_百融.多头.按身份证号查询，近3个月在银行机构最大月申请次数',
        'woe_同盾智信分.小金分.小额现金贷分',
        'woe_聚信利.手机核查信息.近6月话费消费平均值',
        'woe_芝麻分',
        'woe_支付结算协会.收到借款申请机构总数（6个月）',
        'woe_新颜.负面拉黑.睡眠机构数',
        'woe_宜信.违约概率',
        'woe_51公积金.分析得到的月缴额',
        'woe_畅快贷.通讯录.通讯录联系人总个数',
        'woe_蜜罐.180天内历史查询.现金贷查询机构数',
        'woe_宜信.查询时间为三个月内其他机构查询次数',
        'woe_蜜罐.主动联系的种等亲密联系人数',
        'woe_蜜罐.主动联系的亲密联系人数','埋点类.OCR.年龄',
        '百融.多头.按身份证号查询，近3个月在银行机构最大月申请次数',
        '同盾智信分.小金分.小额现金贷分',
        '聚信利.手机核查信息.近6月话费消费平均值',
        '芝麻分',
        '支付结算协会.收到借款申请机构总数（6个月）',
        '新颜.负面拉黑.睡眠机构数',
        '宜信.违约概率',
        '51公积金.分析得到的月缴额',
        '畅快贷.通讯录.通讯录联系人总个数',
        '蜜罐.180天内历史查询.现金贷查询机构数',
        '宜信.查询时间为三个月内其他机构查询次数',
        '蜜罐.主动联系的种等亲密联系人数',
        '蜜罐.主动联系的亲密联系人数','m_score','z']]
debug = debug.T






	
	
	
	
	

	











