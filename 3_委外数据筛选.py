# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 15:22:14 2018

@author: wj61032
"""

import numpy as np
import pandas as pd
from impala.dbapi import connect
from impala.util import as_pandas
from datetime import datetime

#数据地址
path = r'\\file.tounawang.net\风控数据共享\llll\D_HE\1_车贷委外名单筛选\data'+'\\'

def qurry_hdp(sql,flag=1):
    '''
    python直连hive代码
    select * 时传入flag=0,其他时候无需传入flag
    '''
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

'''提取押品表信息'''
sql0 = '''
select
    contractno,
    pawntype,
    pawnsituation,
    carstatus,
    carsituation,
    handletime,
    createtime
FROM ods_lms.tb_lb_pawncarsituation
'''
data0 = qurry_hdp(sql0)
# 保留每个合同最近一条记录
data0.sort_values(['contractno','handletime'],inplace=True)
data0.drop_duplicates(['contractno'],inplace=True,keep='last')

'''读取进度逾期表'''
data1 = pd.read_csv(path+'Tbdsche_over20190829.csv',engine='python') 
#列名统一小写后取需要数据
data1.columns = [x.lower() for x in data1.columns]
data1 = data1[['contractno','inlibrary','loantype','dkfs','loanamt','auditamt'
              ,'returnamt','noreturnamt','returncapital','noreturncapital'
              ,'totalduedays','returnstatus','currentduedays','situation'
              ,'instatus', 'department', 'clear_mode','custname','idcard',
              'phoneno','withdrawdate','sitename','carno_fcdz','carno_md5']]

# 剔除该部分防止合同号相同（仅保留车信易贷）
data1 = data1[~data1['loantype'].isin(['车易贷','信易贷'])]
# 押品表与进度逾期表连表
data = pd.merge(data0,data1,on='contractno') 

#'''押品表中没有的数据'''
#debug = data1[~data1.contractno.isin(data.contractno)]
#debug.currentduedays.value_counts()
#debug.loantype.value_counts()

# 规则0参数：当前逾期天数>15
low_days = 15
# 规则1参数：产品类型
loantypes = ['车贷','车速贷','车信易贷','飞车贷','过户车贷','秒贷','闪贷','随借随还-车贷']
# 规则2参数：未还本金>5000
low_norc = 5000
# 规则3参数：不在库
inlib = '不在库'
# 规则4参数：卖车情况-6.其他、9.人车消失、14.空、16.临时押车出库、21.车辆二押
# 24.正常押证不在库、26.临时押车转押证出库
carsit = [6,9,14,16,21,24,26]
# 规则5参数：抵押情况-已办理
pawnsit = 1
#车辆情况：1:正常押车入库,2:贷后收车入库,
#3:售出处置出库,4:结清赎车出库,
#5:检查出库,6:其他,7:反押入库,
#8:公司赎车入库,9:人车消失,
#10:委外收车入库,11:处置退车入库,
#12:客户主动入库,13:押车待入库,14:空,
#15:临时押车入库,16:临时押车出库,
#17:第三方收车入库,18:审批归还出库,
#19:贷后强制结清出库,21:车辆二押,
#22:临时催收用车归还入库,23:临时催收用车出库,
#24:正常押证不在库,25:正常押证结清,26:临时押车转押证出库

# 根据规则筛选
data_0 = data[data['currentduedays']>low_days] #逾期天数>15
data_1 = data_0[data_0['loantype'].isin(loantypes)] #产品类型
data_2 = data_1[(data_1['noreturncapital']>low_norc)]
data_3 = data_2[data_2['inlibrary']==inlib] #不在库
data_4 = data_3[data_3['carsituation'].isin(carsit)] # 剔除正常出库和在库
data_5 =data_4[data_4['pawnsituation']==pawnsit] #已办理抵押

data_out = data_5[['contractno','sitename','custname','withdrawdate',
                   'auditamt','loantype','currentduedays','noreturncapital',
                   'idcard','phoneno','carno_md5']]
data_out.sort_values(['currentduedays'],inplace=True)
data_out.index = range(1,len(data_out)+1)
data_out.rename(columns={'contractno':'合同编号',
                         'sitename':'服务商',
                         'custname':'客户姓名',
                         'withdrawdate':'放款日期',
                         'auditamt':'RE',
                         'currentduedays':'当前逾期天数',
                         'noreturncapital':'未还本金',
                         'idcard':'身份证号',
                         'phoneno':'手机号',
                         'carno_md5':'加密车牌号'},inplace=True)
data_out['放款日期'] = data_out['放款日期'].apply(lambda x: datetime.strptime(x,'%d%b%Y'))

#输出数据
out_fin0 = data_out[['合同编号', '服务商', '客户姓名', '放款日期', 
                     'RE', 'loantype', '当前逾期天数', '未还本金']]
out_fin0['逾期天数分段'] = pd.cut(out_fin0['当前逾期天数'],
        [15,60,90,180,365,np.inf],
        labels=['(15天,60天]','(60天,90天]','(90天,180天]',
    '(180天,365天]','365天以上'])
print(out_fin0['逾期天数分段'].value_counts())

'''
添加委外解密数据
===============================================================================
'''
other_data = pd.read_excel(path+'委外数据解密_20190830.xlsx')
other_data.rename(columns={'合同号':'合同编号'},inplace=True)
#委外数据连接敏感数据
out_fin1 = pd.merge(out_fin0,other_data,on='合同编号',how='left')
out_fin1 = out_fin1[['合同编号', '服务商',
       '姓名', '证件号码', '手机号', '车牌号', '车架号', '发动车号', '车型型号', 
       '品牌型号', '放款日期', 'RE', '当前逾期天数', '未还本金',
       '逾期天数分段']]
'''
添加彭燕GPS数据
===============================================================================
'''
data_py= pd.read_excel(path+'201909GPS信息提取.xlsx')
data_py.columns
#data_py['找哪最后定位位置']
a = '北京市;天津市;上海市;重庆市;河北省;河南省;云南省;辽宁省;黑龙江省;湖南省;安徽省;山东省;新疆维吾尔;江苏省;浙江省;江西省;湖北省;广西壮族;甘肃省;山西省;内蒙古;陕西省;吉林省;福建省;贵州省;广东省;青海省;西藏;四川省;宁夏回族;海南省;台湾省;香港;澳门;其他'
b = a.replace("市",'').replace('省','').replace("维吾尔",'').replace('回族','').replace('壮族','').split(';')
def prov(x):
    for i in b:
        if i in str(x):
            return i
data_py['找哪设备地区'] =  data_py['找哪定位地址'].apply(prov)  
data_py['天易设备地区'] =  data_py['天易定位地址'].apply(prov)  
data_py['设备地区'] = np.where(data_py['找哪设备地区'].notnull(),
       data_py['找哪设备地区'],
       data_py['天易设备地区'])

data_py.drop_duplicates(['contractno'],inplace=True)

#委外数据与gps数据组合
out_fin1 = pd.merge(out_fin1,data_py,left_on='合同编号',
                    right_on='contractno',how='left')

'''
数据集拆分
中期和高期分别分组
===============================================================================
'''
#高期
out_fin60 = out_fin1[out_fin1['当前逾期天数']>60]
out_fin60.index = range(len(out_fin60))
#中期
out_fin16 = out_fin1[out_fin1['当前逾期天数']<=60]
out_fin16.index = range(len(out_fin16))

#1. 高期历史客户按照25:25:50进行随机分组。
shuffle_index = np.random.permutation(len(out_fin60))
flag0 = int(0.25*len(shuffle_index))
flag1 = int(0.5*len(shuffle_index))

f0 = shuffle_index[:flag0]
f1 = shuffle_index[flag0:flag1]
f2 = shuffle_index[flag1:]

out1 = out_fin60.iloc[f0,]
out2 = out_fin60.iloc[f1,]
out3 = out_fin60.iloc[f2,] 
len(list(out1['合同编号'])+\
    list(out2['合同编号'])+\
    list(out3['合同编号']))

#2. 中期历史客户按照5:5进行随机分组。
shuffle_index = np.random.permutation(len(out_fin16))
flag0 = int(0.5*len(shuffle_index))

f0 = shuffle_index[:flag0]
f1 = shuffle_index[flag0:]

b_out1 = out_fin16.iloc[f0,]
b_out2 = out_fin16.iloc[f1,]
len(list(b_out1['合同编号'])+list(b_out2['合同编号']))

'''结果输出'''
outshow = pd.DataFrame()
outshow['序号'] = ['[61,inf)_25%组',\
'[61,inf)_25%组',\
'[61,inf)_50%组','总计']
outshow['数目'] = [len(out1['未还本金']),len(out2['未还本金']),
       len(out3['未还本金']),len(out_fin60['未还本金'])]
outshow['占比'] = outshow['数目'] / len(out_fin60)
outshow['平均RE'] = [out1['RE'].mean(),out2['RE'].mean(),
        out3['RE'].mean(),out_fin60['RE'].mean()]
outshow['平均未还本金'] = [out1['未还本金'].mean(),out2['未还本金'].mean(),
        out3['未还本金'].mean(),out_fin60['未还本金'].mean()]

outshow['平均当前逾期天数'] = [out1['当前逾期天数'].mean(),out2['当前逾期天数'].mean(),
        out3['当前逾期天数'].mean(),out_fin60['当前逾期天数'].mean()]


outshow1 = pd.DataFrame()
outshow1['序号'] = ['[16,60]_50%组','[16,60]_50%组','总计']
outshow1['数目'] = [len(b_out1['未还本金']),len(b_out2['未还本金']),
       len(out_fin16['未还本金'])]
outshow1['占比'] = outshow1['数目'] / len(out_fin16)
outshow1['平均RE'] = [b_out1['RE'].mean(),b_out2['RE'].mean(),
        out_fin16['RE'].mean()]
outshow1['平均未还本金'] = [b_out1['未还本金'].mean(),b_out2['未还本金'].mean(),
        out_fin16['未还本金'].mean()]
outshow1['平均当前逾期天数'] = [b_out1['当前逾期天数'].mean(),b_out2['当前逾期天数'].mean(),
        out_fin16['当前逾期天数'].mean()]

out1.sort_values(['当前逾期天数'],inplace=True)
out2.sort_values(['当前逾期天数'],inplace=True)
out3.sort_values(['当前逾期天数'],inplace=True)
b_out1.sort_values(['当前逾期天数'],inplace=True)
b_out2.sort_values(['当前逾期天数'],inplace=True)

#筛选流程数据
show0_ = pd.DataFrame([['当前逾期天数>15',data_0.shape[0]],
                        ["产品类型：'车贷','车速贷','车信易贷','飞车贷','过户车贷','秒贷','闪贷','随借随还-车贷'",data_1.shape[0]],
                        ['未还本金>5000',data_2.shape[0]],
                        ['在库情况：不在库',data_3.shape[0]],
                        ["卖车情况：6.其他、9.人车消失、14.空、16.临时押车出库、21.车辆二押、24.正常押证不在库、26.临时押车转押证出库",data_4.shape[0]],
                        ["抵押情况：已办理",data_5.shape[0]]],
                        columns=['流程','数目'])
#结果输出
writer = pd.ExcelWriter(path+'各组数据详情_{}.xlsx'.format(datetime.today().date().strftime('%Y%m%d')))
show0_.to_excel(writer,'数据筛选流程',index=False)
outshow.to_excel(writer,'高逾各组统计数据',index=False)
outshow1.to_excel(writer,'中逾各组统计数据',index=False)
out1.to_excel(writer,'序号1_25%高期数据',index=False)
out2.to_excel(writer,'序号2_25%高期数据',index=False)
out3.to_excel(writer,'序号3_50%高期数据',index=False)
b_out1.to_excel(writer,'序号4_50%中期数据',index=False)
b_out2.to_excel(writer,'序号5_50%中期数据',index=False)
writer.save()






'''进一步验证'''
#debug = pd.read_excel(path+'各组数据详情_{}.xlsx'.format(datetime.today().date().strftime('%Y%m%d')),
#                      encoding='gbk',sheetname=[0,1,2,3,4])
#debug1 = pd.concat([debug[0],debug[1],debug[2],debug[3],debug[4]])
#print(debug1['合同编号'].value_counts())











