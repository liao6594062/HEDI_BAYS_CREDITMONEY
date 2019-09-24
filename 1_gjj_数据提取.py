# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 13:48:28 2019

@author: wj61032
"""

import pandas as pd
import numpy as np
from datetime import datetime

'''数据载入及查看'''
data_path = r'\\file.tounawang.net\风控数据共享\llll\D_HE\2_51benchmark模型\51公积金分析20190726\data'+'\\'

'''逾期数据'''
repay_data = pd.read_csv(data_path+'51公积金逾期数据.csv',engine='python')
repay_data['首逾4+'] = np.where(repay_data['fpd']>=4,1,0)
repay_data['最大逾期4+'] = np.where(repay_data['cpd_max']>=4,1,0)
repay_data['放款年月'] = repay_data['payment_time'].apply(lambda x: datetime.strptime(str(x)[:9],'%d%b%Y').strftime('%Y-%m'))
repay_data['first_returndate']=repay_data['first_returndate'].apply(lambda x: datetime.strptime(str(x)[:9],'%d%b%Y'))
repay_data['payment_time']=repay_data['payment_time'].apply(lambda x: datetime.strptime(str(x)[:18],'%d%b%Y:%H:%M:%S'))
repay_data = repay_data[repay_data['first_returndate']<=datetime(2019,7,25)]
repay_data['最大逾期7+'] = np.where(repay_data['cpd_max']>=7,1,0)

'''连接决策引擎数据'''
from impala.dbapi import connect
from impala.util import as_pandas

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

request_ids = ""
for i in repay_data.request_id:
    request_ids+="'"+i+"',"
request_ids = request_ids[:-1]

sql ="""
select * from provider_api_db.tio_decision_credit_auto_merge
where data_date>='2019-05-01'
and ck_app_node in ('CKD_BASIC_CERT','CKD_MANUAL_CERT','CKD_ADVANCED_CERT')
and request_id in ({})
""".format(request_ids)

deci_data = qurry_hdp(sql,0)

#数据合并
repay_cols = ['contract_no','cust_id','request_id',
              'fpd', 'cpd_max','currentduedays', 'firstreturndate',
       'noreturncapital','放款年月','payment_time','loan_amt']
df = pd.merge(repay_data[repay_cols],deci_data,on='request_id')
df = df[df['last_update_date']<=df['payment_time']]
df = df.sort_values(['contract_no','last_update_date']).drop_duplicates(['contract_no'],keep='last')

'''定义y'''
#坏客户定义：至少还款3期+4天表现期，最大逾期天数≥4
df['y'] = np.where(df['cpd_max']>=4,1,0)
df['额度_分段'] = pd.cut(df['creditda_cline'],[0,5000,6000,7000,8000,9000,10000,
  11000,12000,13000,14000,15000,20000,50000,100000,np.inf],right=False)
df['放款金额_分段'] = pd.cut(df['loan_amt'],[0,5000,6000,7000,8000,9000,10000,
  11000,12000,13000,14000,15000,20000,50000,100000,np.inf],right=False)
df['y'].groupby(df['额度_分段']).agg(['count','sum','mean'])
df['y'].groupby(df['放款金额_分段']).agg(['count','sum','mean'])

X = df.copy()

'''主键保护，方便后续连表，添加字符防止后续代码转换为float'''
X['contract_no'] = X['contract_no'].astype(str)+'保'
X.select_dtypes(exclude=['category']).to_excel(data_path+'X.xlsx',index=False)









