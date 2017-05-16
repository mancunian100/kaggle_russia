#这个文件主要对缺失值进行了处理，少的直接用平均值进行填充，缺的多的用xgboost进行预测

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mode

from sklearn import model_selection, preprocessing
import xgboost as xgb
import datetime
#now = datetime.datetime.now()
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import make_scorer

pd.options.mode.chained_assignment = None  # default='warn'
pd.set_option('display.max_columns', 500)

train = pd.read_csv('kaggle/russia/train.csv')
test = pd.read_csv('kaggle/russia/test.csv')
macro = pd.read_csv('kaggle/russia/macro.csv')


y_train = train["price_doc"]
x_train = train.drop(["id", "timestamp", "price_doc"], axis=1)
x_test = test.drop(["id", "timestamp"], axis=1)
id_test = test.id

# can't merge train with test because the kernel run for very long time
dtype_df=x_train.dtypes.reset_index()
dtype_df.columns=["Count","ColumnsType"]
dtype_df.groupby("ColumnsType").aggregate("count").reset_index()
#统计属性为object的值
col_object=[]
for x in  dtype_df[dtype_df['ColumnsType']==object]['Count']:
    col_object.append(x)
#统计的结果 int64 共有155，float64共有119 object 15
#缺失值统计
missing_df=train.isnull().sum(axis=0).reset_index()
missing_df.columns=['columns_name','missing_count']
missing_df=missing_df.ix[missing_df['missing_count']>0]
missing_df['missing_ratio']=missing_df['missing_count']/30471

missing_df1=test.isnull().sum(axis=0).reset_index()
missing_df1.columns=['columns_name','missing_count']
missing_df1=missing_df1.ix[missing_df1['missing_count']>0]
missing_df1['missing_ratio']=missing_df1['missing_count']/7662

for x in missing_df['columns_name']:
    # print(x,">>",x_test[x].dtype)
    if x_test[x].dtype == 'int64':
        print(x)

for x in x_test.columns:
    if x_test[x].dtype == 'object':
        print(x)
l1=[]
l2=[]
for x in missing_df['columns_name']:
    l1.append(x)
for x in missing_df1['columns_name']:
    l2.append(x)
[x for x in l1 if x not in l2]

ind=np.arange(missing_df.shape[0])
width=0.9
fig, ax = plt.subplots(figsize=(12,18))
rects = ax.barh(ind, missing_df.missing_count.values, color='y')
ax.set_yticks(ind)
ax.set_yticklabels(missing_df.columns_name.values, rotation='horizontal')
ax.set_xlabel("Count of missing values")
ax.set_title("Number of missing values in each column")
plt.show(block=False)

# data=train.groupby(['Title','Pclass'])['age']
# train['age']=data.transform(lambda x:x.fillna())
#
#------------------------------------------
#处理缺失值
#--------------------------------------------

full=pd.concat([train,test])
#state中有一个异常值33替换为3
full['state'].replace(33,3,inplace=True)
#从scipy导入的求众数的函数,注意不能有缺失值
#填充test中缺少的product_type和green_part_2000
mode(full['green_part_2000']).mode[0]
full['green_part_2000'].fillna(full['green_part_2000'].mean(),inplace=True)
full['product_type'].value_counts()
full['product_type'].fillna('Investment',inplace=True)
#对build_year进行处理
full.build_year.value_counts()
full.loc[full['build_year']==20052009,'build_year']=2005
full.loc[full['build_year']==0,'build_year']=np.nan
full.loc[full['build_year']==1,'build_year']=np.nan
full.loc[full['build_year']==20,'build_year']=2000
full.loc[full['build_year']==215,'build_year']=2015
full.loc[full['build_year']==3,'build_year']=np.nan
full.loc[full['build_year']==2,'build_year']=np.nan
full.loc[full['build_year']==71,'build_year']=np.nan
full.loc[full['build_year']==4965,'build_year']=np.nan
#对sub_area进行重新划分，这个脚本里面没有，再dataprocess1.py中分块了，效果不是很好

# full.loc[full['sub_area']=='']

full.drop(["id", "timestamp", "price_doc"], axis=1,inplace=True)



#之前是289列，get_dummies之后是451列
full=pd.get_dummies(full,columns=col_object)


#模型调参
def get_model(estimator, parameters, X_train, y_train, scoring):
    model = GridSearchCV(estimator, param_grid=parameters, scoring=scoring)
    model.fit(X_train, y_train)
    return model.best_estimator_

#
# X=full[full.floor.notnull()].drop('floor',axis=1)
# y=full[full.floor.notnull()].floor
#
# X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=2017)
# XGB = xgb.XGBRegressor(max_depth=4, seed= 2017)
scoring = make_scorer(mean_absolute_error, greater_is_better=False)
parameters = {'reg_alpha':np.linspace(0.1,1.0,5), 'reg_lambda': np.linspace(1.0,3.0,5)}
# reg_xgb = get_model(XGB, parameters, X_train, y_train, scoring)
# print (reg_xgb)
# print ("Mean absolute error of test data: {}".format(mean_absolute_error(y_test, reg_xgb.predict(X_test))))
# #3.49323
# pred = reg_xgb.predict(full[full.floor.isnull()].drop('floor', axis=1))
# full.loc[(full.floor.isnull()),'floor']=pred
for att in missing_df['columns_name']:
    print(train[att].isnull().sum())

for att in missing_df['columns_name']:
    X = full[full[att].notnull()].drop(att, axis=1)
    y = full[full[att].notnull()][att]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2017)

    XGB = xgb.XGBRegressor(max_depth=4, seed=2017)

    reg_xgb = get_model(XGB, parameters, X_train, y_train, scoring)
    print(reg_xgb)
    print(att,"Mean absolute error of test data: {}".format(mean_absolute_error(y_test, reg_xgb.predict(X_test))))
    pred = reg_xgb.predict(full[full[att].isnull()].drop(att, axis=1))
    full.loc[(full[att].isnull()), att] = pred



full.to_csv('kaggle/russia/full.csv', index=False)