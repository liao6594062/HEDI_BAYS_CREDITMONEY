# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 11:27:30 2019

@author: wj61032
"""
import sys
sys.path.append(r'\\file.tounawang.net\风控数据共享\小工具\internal_lib\treecut\scripts')
# 以上设置请勿修改
import pandas as pd
import numpy as np
from tree_cut import *
from datetime import datetime
import matplotlib.pyplot as plt

data_path=r'\\file.tounawang.net\风控数据共享\llll\D_HE\2_51benchmark模型\51公积金分析20190726\data'+'\\'
X=pd.read_excel(data_path+'X.xlsx')

'''数据质量察看'''
X.replace({-99:np.nan,
           '':np.nan,
           '-99':np.nan,
           '-99.0':np.nan},inplace=True)
isnull_ = X.groupby('放款年月').apply(lambda x: x.isnull().mean())
isnull_.to_excel(data_path+'分客群_变量缺失率.xlsx')

# 缺失率分布
def null_rate(df):
    def null_rate(x):
        if x == 0:
            y = 'a_[0%]'
        elif x<=0.1:
            y = 'b_(0%,10%]'
        elif x<=0.2:
            y = 'c_(10%,20%]'
        elif x<=0.3:
            y = 'd_(20%,30%]'
        elif x<=0.4:
            y = 'e_(30%,40%]'
        elif x<=0.5:
            y = 'f_(40%,50%]'
        elif x<=0.6:
            y = 'g_(50%,60%]'
        elif x<=0.7:
            y = 'h_(60%,70%]'
        elif x<=0.8:
            y = 'i_(70%,80%]'
        elif x<=0.9:
            y = 'j_(80%,90%]'
        elif x<1:
            y = 'k_(90%,100%)'
        elif x==1:
            y = 'l_[100%]'
        return y
        
    null_data = pd.DataFrame(df.drop(columns=['y']).isnull().mean(),
                             columns=['缺失率'])
    null_data['缺失率分段'] = null_data['缺失率'].apply(null_rate)   
    null_data1 = pd.DataFrame(null_data['缺失率分段'].value_counts().sort_index())
    null_data1.columns=['特征数量']
    null_data1['占比'] = null_data1['特征数量']/null_data1['特征数量'].sum()
    null_data1.index.name='缺失比例'
    return null_data1
  
null_rate_res = null_rate(X)

'''变量类型查看'''
#变量类型转换
print(X.dtypes.value_counts())
for col in X.select_dtypes(['object']).columns:
    if col =='contract_no':
        continue
    try:
        X[col] = X[col].astype(float)
        print(col,'成功转换为数值型')
    except:
        try:
           X[col] = X[col].astype(np.datetime64)
           print(col,'成功转换为时间型')
        except:
           pass

'''变量名替换'''
col_dict = pd.read_excel(data_path+'dict_.xlsx')
col_dict = dict(zip(col_dict['col_name'],col_dict['label']))
X.rename(columns=col_dict,inplace=True)

#剔除全空变量
X = X.select_dtypes(exclude=['category'])
isnull_1 = X.isnull().mean()
null_col = isnull_1.index[isnull_1==1]
X = X.drop(columns=null_col)

'''与y的相关性查看'''
corr = X.fillna(-99).corr()['y']
corr = np.abs(corr)
corr_y = corr.sort_values(ascending=False)

'''IV查看'''
# 数值型变量
iv_cols = X.select_dtypes(['float64','int64','int32','bool'])
iv_cols = [x for x in iv_cols if x not in ['y',
                                           'currentduedays',
                                           'cpd_max',
                                           'fpd',
                                           '针对双卡双待手机，卡1的IMSI']]

ivs = woe_show(X,iv_cols,data_path,file_name='51公积金单变量.xlsx',
               max_groups=6,target='y')

'''变量筛选'''
def col_show(df,col,max_groups=6,target='y',split=[]):
    woe_res = {}
    temp_res = WOE(df, col, target, max_groups, split=split, special_points=[])
    woe_res[col+'整体'] = temp_res[2]

    '''分月数据查看'''
    temp_res0 = df['y'].groupby([df['woe_'+col],df['放款年月'].apply(lambda x: x.strftime('%Y-%m'))]).count().unstack()
    temp_res0.ix['总计',:] = df['y'].groupby(df['放款年月'].apply(lambda x: x.strftime('%Y-%m'))).count()
    temp_res0 = temp_res0/temp_res0.ix['总计',:]
    temp_res0['原始'] = temp_res0.index
    temp_res0['原始'] = temp_res0['原始'].replace({value:key for key,value in temp_res[0].items()})
    temp_res0.index.name='占比'
    
    temp_res1 = df['y'].groupby([df['woe_'+col],df['放款年月'].apply(lambda x: x.strftime('%Y-%m'))]).mean().unstack()
    temp_res1.ix['总计',:] = df['y'].groupby(df['放款年月'].apply(lambda x: x.strftime('%Y-%m'))).mean()
    temp_res1.index.name='坏账率'
    
    temp_res2 = temp_res1.copy()/temp_res1.ix['总计',:].copy()
    temp_res2.index.name='提升度'
    
    temp_res1['原始'] = temp_res1.index
    temp_res1['原始'] = temp_res1['原始'].replace({value:key for key,value in temp_res[0].items()})
    temp_res2['原始'] = temp_res2.index
    temp_res2['原始'] = temp_res2['原始'].replace({value:key for key,value in temp_res[0].items()})
    
    
    woe_res[col+'占比'] = temp_res0
    woe_res[col+'坏账率'] = temp_res1
    woe_res[col+'提升度'] = temp_res2  
    print(woe_res[col+'整体'],'\n')
    print(woe_res[col+'占比'],'\n')
    print(woe_res[col+'坏账率'],'\n')
    print(woe_res[col+'提升度'],'\n')
    return woe_res

#初选变量
fcols = pd.read_excel(data_path+'51公积金单变量选取.xlsx')
fcols = [x.replace('woe_','') for x in fcols['变量名']]
fcols = [x for x in fcols if x not in ['蜜罐.灰度分']]

Xf = X[fcols+['y','放款年月','contract_no','请求id']]

ivs = woe_show(Xf,fcols,data_path,file_name='初筛单变量20190730.xlsx',max_groups=6,target='y')

Xf.replace({-99:np.nan,
           '':np.nan,
           '-99':np.nan,
           '-99.0':np.nan},inplace=True)

'''分箱合并'''
tot_res = {}
col='同业专家卡.总分'
split = [496.5]
show = col_show(Xf,col,max_groups=6,target='y',split=split)
tot_res[col] = show
Xf['woe_'+col].value_counts()

col='埋点类.OCR.年龄'
split = [39.5]
show = col_show(Xf,col,max_groups=6,target='y',split=split)
tot_res[col] = show
Xf['woe_'+col].value_counts()

col='蜜罐.社交影响力'
split = [98.395]
show = col_show(Xf,col,max_groups=6,target='y',split=split)
tot_res[col] = show
Xf['woe_'+col].value_counts()
Xf['woe_'+col].replace({np.inf:-0.1989},inplace=True)

col='同盾智信分.小金分.小额现金贷分'
split = [484.5,662.5]
show = col_show(Xf,col,max_groups=6,target='y',split=split)
tot_res[col] = show
Xf['woe_'+col].value_counts()

col='51公积金.公司月缴额'
split = [472.75]
show = col_show(Xf,col,max_groups=6,target='y',split=split)
tot_res[col] = show
Xf['woe_'+col].value_counts()
Xf['woe_'+col].replace({-0.304475:-0.2500},inplace=True)

col='聚信立.通讯列表手机号码总个数'
split = [86.5]
show = col_show(Xf,col,max_groups=6,target='y',split=split)
tot_res[col] = show
Xf['woe_'+col].value_counts()

col='51公积金.分析得到的月缴额'
split = [905.5]
show = col_show(Xf,col,max_groups=6,target='y',split=split)
tot_res[col] = show
Xf['woe_'+col].value_counts()
Xf['woe_'+col].replace({-0.600741:-0.2423},inplace=True)

col='聚信利.手机核查信息.近6月主叫次数平均值'
split = [8.585]
show = col_show(Xf,col,max_groups=6,target='y',split=split)
tot_res[col] = show

col='蜜罐.一阶联系人总数'
split = [131.5]
show = col_show(Xf,col,max_groups=6,target='y',split=split)
tot_res[col] = show
Xf['woe_'+col].value_counts()
Xf['woe_'+col].replace({np.inf:-0.1748},inplace=True)

col='聚信利.手机核查信息.近6月话费消费平均值'
split = [43.235]
show = col_show(Xf,col,max_groups=6,target='y',split=split)
tot_res[col] = show
Xf['woe_'+col].value_counts()

col='畅快贷.通讯录.通讯录联系人总个数'
split = [622]
show = col_show(Xf,col,max_groups=6,target='y',split=split)
tot_res[col] = show
Xf['woe_'+col].value_counts()

col='百融.多头.按身份证号查询，近3个月在非银机构最大月申请次数'
Xf[col] = Xf[col].replace('',np.nan).astype(float)
split = [16.5]
show = col_show(Xf,col,max_groups=6,target='y',split=split)
tot_res[col] = show
Xf['woe_'+col].value_counts()
Xf['woe_'+col].replace({np.inf:-0.3547},inplace=True)

col='百融.多头.按身份证号查询，近6个月最大月申请次数'
Xf[col] = Xf[col].replace('',np.nan).astype(float)
split = [15.5]
show = col_show(Xf,col,max_groups=6,target='y',split=split)
tot_res[col] = show
Xf['woe_'+col].value_counts()
Xf['woe_'+col].replace({np.inf:-0.4515},inplace=True)

col='新颜.负面拉黑.睡眠机构数'
split = [4.5]
show = col_show(Xf,col,max_groups=6,target='y',split=split)
tot_res[col] = show
Xf['woe_'+col].value_counts()

col='百融.多头.按身份证号查询，近3个月最大月申请次数'
Xf[col] = Xf[col].replace('',np.nan).astype(float)
split = [15.5]
show = col_show(Xf,col,max_groups=6,target='y',split=split)
tot_res[col] = show
Xf['woe_'+col].value_counts()
Xf['woe_'+col].replace({np.inf:-0.4715},inplace=True)

col='百融.多头.按手机号查询，近3个月最大月申请次数'
Xf[col] = Xf[col].replace('',np.nan).astype(float)
split = [16.5]
show = col_show(Xf,col,max_groups=6,target='y',split=split)
tot_res[col] = show
Xf['woe_'+col].value_counts()
Xf['woe_'+col].replace({np.inf:-0.5086},inplace=True)

col='宜信.查询时间为三个月内其他机构查询次数'
split = [4.5]
show = col_show(Xf,col,max_groups=6,target='y',split=split)
tot_res[col] = show
Xf['woe_'+col].value_counts()
Xf['woe_'+col].replace({np.inf:-0.2199},inplace=True)

col='蜜罐.主动联系的种等亲密联系人数'
split = [64.5]
show = col_show(Xf,col,max_groups=6,target='y',split=split)
tot_res[col] = show
Xf['woe_'+col].value_counts()
Xf['woe_'+col].replace({np.inf:-0.1030},inplace=True)

col='蜜罐.主动联系的亲密联系人数'
split = [33.5]
show = col_show(Xf,col,max_groups=6,target='y',split=split)
tot_res[col] = show
Xf['woe_'+col].value_counts()
Xf['woe_'+col].replace({np.inf:-0.0786},inplace=True)

col = '百融.多头.按手机号查询，近6个月在银行机构申请最小间隔天数'
Xf[col] = Xf[col].replace('',np.nan).astype(float)
Xf['百融.多头.按手机号查询，近6个月在银行机构申请最小间隔天数[0,10.5]'] = \
np.where((Xf['百融.多头.按手机号查询，近6个月在银行机构申请最小间隔天数']<11)&\
         (Xf['百融.多头.按手机号查询，近6个月在银行机构申请最小间隔天数']>=0),1,0)
col='百融.多头.按手机号查询，近6个月在银行机构申请最小间隔天数[0,10.5]'
split = [0.5]
show = col_show(Xf,col,max_groups=6,target='y',split=split)
tot_res[col] = show
Xf['woe_'+col].value_counts()

col='芝麻分'
Xf[col] = Xf[col].replace('',np.nan).astype(float)
split = [701.5,717.5]
show = col_show(Xf,col,max_groups=6,target='y',split=split)
tot_res[col] = show
Xf['woe_'+col].value_counts()
Xf['woe_'+col].replace({np.inf:-0.2263},inplace=True)

col='支付结算协会.收到借款申请机构总数（6个月）'
Xf[col] = Xf[col].replace('',np.nan).astype(float)
split = [0.5]
show = col_show(Xf,col,max_groups=6,target='y',split=split)
tot_res[col] = show
Xf['woe_'+col].value_counts()

col='聚信利.手机核查信息.夜间活动比例'
split = [0.065]
show = col_show(Xf,col,max_groups=6,target='y',split=split)
tot_res[col] = show
Xf['woe_'+col].value_counts()

col = '百融.多头.按身份证号查询，近3个月在银行机构平均每月申请次数(有申请月份平均)'
Xf[col] = Xf[col].replace('',np.nan).astype(float)
Xf['百融.多头.按身份证号查询，近3个月在银行机构平均每月申请次数(有申请月份平均)>1.835'] = \
np.where((Xf['百融.多头.按身份证号查询，近3个月在银行机构平均每月申请次数(有申请月份平均)']>1.835),1,0)
col='百融.多头.按身份证号查询，近3个月在银行机构平均每月申请次数(有申请月份平均)>1.835'
split = [0.5]
show = col_show(Xf,col,max_groups=6,target='y',split=split)
tot_res[col] = show
Xf['woe_'+col].value_counts()

col='鹏元.学历'
Xf[col] = Xf[col].replace('',np.nan).astype(float)
split = [3.5]
show = col_show(Xf,col,max_groups=6,target='y',split=split)
tot_res[col] = show
Xf['woe_'+col].value_counts()

col='蜜罐.180天内历史查询.现金贷查询机构数'
split = [5.5]
show = col_show(Xf,col,max_groups=6,target='y',split=split)
tot_res[col] = show

Xf['宜信.违约概率>0.2'] = \
np.where((Xf['宜信.违约概率']>0.2),1,0)
col='宜信.违约概率>0.2'
split = [0.5]
show = col_show(Xf,col,max_groups=6,target='y',split=split)
tot_res[col] = show

col = '百融.多头.按身份证号查询，近3个月在银行机构最大月申请次数'
Xf[col] = Xf[col].replace('',np.nan).astype(float)
Xf['百融.多头.按身份证号查询，近3个月在银行机构最大月申请次数>1.5'] = \
np.where((Xf['百融.多头.按身份证号查询，近3个月在银行机构最大月申请次数']>1.5),1,0)
col='百融.多头.按身份证号查询，近3个月在银行机构最大月申请次数>1.5'
split = [0.5]
show = col_show(Xf,col,max_groups=6,target='y',split=split)
tot_res[col] = show

col='蜜罐.是否购买理财产品'
split = [0.5]
show = col_show(Xf,col,max_groups=6,target='y',split=split)
tot_res[col] = show
Xf['woe_'+col].value_counts()

col='宜信.最近六个月内审批结果为拒绝的借款记录条数'
split = [0.5]
show = col_show(Xf,col,max_groups=6,target='y',split=split)
tot_res[col] = show
Xf['woe_'+col].value_counts()
Xf['woe_'+col].replace({np.inf:-0.3150},inplace=True)

'''共线性及建模'''
f_X = ['woe_'+ x for x in list(tot_res.keys())]
X_fin = Xf[f_X]
y = Xf['y']
corr = X_fin.corr()

data_vif = X_fin
from sklearn import linear_model
vif = pd.DataFrame(columns=['变量名','vif'])
for col in X_fin.columns:
    print('正在评估：',col,'\n')
    reg = linear_model.LinearRegression()
    reg.fit(data_vif.drop(columns=[col]),data_vif[col])
    vifi = 1/(1-reg.score(data_vif.drop(columns=[col]),data_vif[col]))
    vif = vif.append(pd.DataFrame([[col,vifi]],columns=['变量名','vif']))
vif.sort_values(['vif'],ascending=False,inplace=True)
vif.index = range(1,len(vif)+1)


def vif_show(train,t_col):
    vif_col = [i for i in f_X if i != t_col]
    vif_drop = pd.DataFrame(columns=['剔除的变量名','vif_of_'+t_col])
    for i in vif_col:
        reg = linear_model.LinearRegression()
        temp_col =[x for x in vif_col if x != i]
        reg.fit(train[temp_col],train[t_col])
        vifi = 1/(1-reg.score(train[temp_col],train[t_col]))
        vif_drop = vif_drop.append(pd.DataFrame([[i,vifi]],columns=['剔除的变量名','vif_of_'+t_col]))
    vif_drop.sort_values('vif_of_'+t_col,inplace=True)
    return vif_drop

v0 = vif_show(X_fin,'woe_蜜罐.一阶联系人总数')

vif_drop0 = tot_res['百融.多头.按身份证号查询，近3个月最大月申请次数']
vif_drop1 = tot_res['百融.多头.按身份证号查询，近6个月最大月申请次数']
vif_drop2 = tot_res['51公积金.公司月缴额']
vif_drop3 = tot_res['51公积金.分析得到的月缴额']
vif_drop4 = tot_res['百融.多头.按手机号查询，近3个月最大月申请次数']
vif_drop5 = tot_res['百融.多头.按身份证号查询，近3个月最大月申请次数']


#共线性剔除变量
f_X = [x for x in f_X if x not in ['woe_百融.多头.按身份证号查询，近6个月最大月申请次数',
                                   'woe_51公积金.公司月缴额',
                                   'woe_百融.多头.按手机号查询，近3个月最大月申请次数',
                                   'woe_同业专家卡.总分']]

X_fin = Xf[f_X]
y = Xf['y']
corrX = X_fin.corr() 

#剩余变量排序
selec_cols = [x.replace('woe_','') for x in f_X]
imp = []
for col in selec_cols:
    imp.append([col,tot_res[col][col+'整体'].ix['All','IV']])
imp = pd.DataFrame(imp,columns=['变量名','IV']).sort_values(['IV'],ascending=False)
imp['变量名']=imp['变量名'].apply(lambda x: 'woe_'+x)

'''变量筛选'''
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split,cross_validate
from sklearn.metrics import make_scorer,accuracy_score,roc_curve,roc_auc_score,recall_score

def feature_selection(imp,X,y,scoring='auc'):
    #输出结果
    feature_selection_ = pd.DataFrame(columns=['变量名',
                                              '变量集合',
                                              '变量数目',
                                              'mean-'+scoring,
                                              'std-'+scoring,
                                              '是否保留该变量',
                                              '最终变量集'])
    #设置初始分数和初始集合，分数为越大越好
    scoring_initial=0
    col_initial = []
    flag=1
    for col in imp['变量名']:
        print('正在评估RANK{0}变量：{1}'.format(str(flag),col))
        if col not in col_initial:
            #使用当前变量集合训练模型
            col_initial.append(col)
            clf = LogisticRegression()
            cv_results = cross_validate(clf,X[col_initial],
                                        y,cv=5,n_jobs=-1,
                                        scoring=['roc_auc'])

            #确定当前变量去留
            is_stay = '保留'
            lag_col = col_initial.copy()
            if np.mean(cv_results['test_roc_auc'])<=scoring_initial:
                col_initial.remove(col)
                is_stay = '不保留'
            else:
                scoring_initial = np.mean(cv_results['test_roc_auc'])
                
            temp_df = pd.DataFrame([[col,lag_col,
             len(lag_col),
             np.mean(cv_results['test_roc_auc']),
             np.std(cv_results['test_roc_auc']),
             is_stay,
             col_initial]],columns=['变量名',
                                              '变量集合',
                                              '变量数目',
                                              'mean-'+scoring,
                                              'std-'+scoring,
                                              '是否保留该变量',
                                              '最终变量集'])
            print("当前scoring:{0},当前scoring_std:{1},'\n'".format(np.mean(cv_results['test_roc_auc']),
                  np.std(cv_results['test_roc_auc'])))
            feature_selection_ = feature_selection_.append(temp_df)
            flag+=1
    return feature_selection_

cols_selection = feature_selection(imp,X_fin,y,scoring='auc')


fin_cols = ['woe_同盾智信分.小金分.小额现金贷分', 'woe_新颜.负面拉黑.睡眠机构数', 'woe_支付结算协会.收到借款申请机构总数（6个月）', 'woe_芝麻分', 'woe_畅快贷.通讯录.通讯录联系人总个数', 'woe_聚信利.手机核查信息.近6月话费消费平均值', 'woe_51公积金.分析得到的月缴额', 'woe_百融.多头.按身份证号查询，近3个月在银行机构最大月申请次数>1.5', 'woe_埋点类.OCR.年龄', 'woe_蜜罐.主动联系的种等亲密联系人数', 'woe_宜信.违约概率>0.2', 'woe_蜜罐.180天内历史查询.现金贷查询机构数', 'woe_蜜罐.主动联系的亲密联系人数', 'woe_宜信.查询时间为三个月内其他机构查询次数']
for col in fin_cols:
    print(col)
    print(Xf[col].value_counts(),'\n\n')

clf = LogisticRegression()
#penalty='l1',C=1
clf.fit(X_fin[fin_cols],y)
pred_p = clf.predict_proba(Xf[fin_cols])[:,1] # 预测为1的概率
fpr, tpr, th = roc_curve(y, pred_p)
ks = max(tpr - fpr)        
auc = roc_auc_score(y, pred_p)  
print(ks,auc)

imp = pd.DataFrame()
imp['变量名'] = Xf[fin_cols].columns
imp['参数估计'] = clf.coef_[0]
imp.sort_values(['参数估计'],inplace=True)
imp.index = range(1,len(imp)+1)

'''
结果查看
===============================================================================
'''
import matplotlib.pyplot as plt
clf = joblib.load(data_path+'模型20190730_KS448.pkl')
Xf['score'] = clf.predict_proba(Xf[fin_cols])[:,1]*100.000
col='score'
split=[]
show = col_show(Xf,col,max_groups=2,target='y',split=split)
tot_res[col] = show

pred_p = clf.predict_proba(X_fin[fin_cols])[:,1]*100.000 # 获取bins
temp = pd.DataFrame()
temp['类别']=y
temp['得分']=pred_p
bins = []
for i in np.arange(0.2,1,0.2):
    bins.append(temp['得分'].quantile(i))
bins = bins[::-1]
bins.append(0)
bins = bins[::-1]
bins.append(100)
bins = sorted(list(set(bins)))

def model_result(y_temp,p_temp,roc_name):
    temp = pd.DataFrame()
    temp['类别']=y_temp
    temp['得分']=p_temp
    temp['得分1'] = pd.cut(temp['得分'],bins,right=False)
    temp0 = temp['得分'].groupby([temp['得分1'],temp['类别']]).count().unstack().sort_index(ascending=False)
    temp0['客户数'] = temp0.sum(axis=1)
    temp0 = temp0.rename(columns={1:'坏客户数'})
    temp0['占比'] = temp0['客户数']/temp0['客户数'].sum()
    temp0['累计占比'] = temp0['客户数'].cumsum()/temp0['客户数'].sum()
    temp0.fillna(0,inplace=True)
    temp0['坏客户比率'] = temp0['坏客户数']/temp0['客户数']
    temp0['坏客户累计'] = temp0['坏客户数'].cumsum()
    temp0['好客户累计'] = (temp0['客户数']-temp0['坏客户数']).cumsum()
    temp0['坏客户累计占比'] = temp0['坏客户累计'] / temp0['坏客户数'].sum()
    temp0['好客户累计占比'] = temp0['好客户累计'] / (temp0['客户数']-temp0['坏客户数']).sum()
    temp0['KS'] = temp0['坏客户累计占比'] - temp0['好客户累计占比']
    
    temp1 = temp0[['客户数','占比','累计占比','坏客户数','坏客户比率',
                   '坏客户累计占比','好客户累计占比','KS']]
    temp1.index.name='得分'
    fpr, tpr, th = roc_curve(y_temp, p_temp)    
    print('ks:',max(tpr - fpr))
    # 绘图
    lw=2
    plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.3f)' % metrics.auc(fpr, tpr))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.02])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC AUC')
    plt.legend(loc="lower right")
    #plt.savefig(r'\\10.0.3.200\分析中心\共享\3.专项分析\1、车贷\8、模型\结清坏客户预测\fig\model_fig'+'\\'+roc_name+'.jpg',dpi=500)
    plt.show()
    return temp1


from sklearn import metrics
pred_p = clf.predict_proba(Xf[fin_cols])[:,1]*100.000
result0 = model_result(y_temp=y, p_temp=pred_p,roc_name='训练集')

'''模型保存'''
from sklearn.externals import joblib
#joblib.dump(clf,data_path+'模型20190730_KS448.pkl')


Xf['分数'] = clf.predict_proba(Xf[fin_cols])[:,1]*100.000
Xf['放款年月'] = Xf['放款年月'].apply(lambda x: x.strftime('%Y-%m'))
Xf['分数分段'] = pd.cut(Xf['分数'],bins)
res = Xf['y'].groupby([Xf['放款年月'],Xf['分数分段']]).agg(['count','sum','mean'])
Xf['放款年月']

X1 = Xf[Xf['放款年月'].isin(['2019-05'])]
X2 = Xf[Xf['放款年月'].isin(['2019-06'])]

fpr, tpr, th = roc_curve(X1['y'], X1['分数'])
ks = max(tpr - fpr)        
auc = roc_auc_score(X1['y'], X1['分数'])  
print(ks,auc)

fpr, tpr, th = roc_curve(X2['y'], X2['分数'])
ks = max(tpr - fpr)        
auc = roc_auc_score(X2['y'], X2['分数'])  
print(ks,auc)

'''入模变量结果输出'''
imp = []
for col in fin_cols:
    col=col.replace('woe_','')
    imp.append([col,tot_res[col][col+'整体'].ix['All','IV']])
imp = pd.DataFrame(imp,columns=['变量名','IV']).sort_values(['IV'],ascending=False)
imp['变量名']=imp['变量名'].apply(lambda x: 'woe_'+x)

show_cols = imp['变量名']

writer = pd.ExcelWriter(data_path+'最终入模变量20190730.xlsx')
startrows=0
for col in fin_cols:
    col = col.replace('woe_','')
    tot_col_show = tot_res[col]
    tot_col_show[col+'整体'].to_excel(writer,startrow=startrows,sheet_name='WOE详细')
    tot_col_show[col+'提升度'].to_excel(writer,startrow=startrows,startcol=9,sheet_name='WOE详细')
    tot_col_show[col+'坏账率'].to_excel(writer,startrow=startrows,startcol=21,sheet_name='WOE详细')
    tot_col_show[col+'占比'].to_excel(writer,startrow=startrows,startcol=32,sheet_name='WOE详细')
    startrows+=len(tot_col_show[col+'整体'])+3
writer.save()
writer.close()

Xf[['contract_no','分数','y','放款年月']].to_excel(data_path+'51gjj_benchmark_score0.xlsx',
  index=False)









