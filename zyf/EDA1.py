import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection, preprocessing
import xgboost as xgb


pd.options.mode.chained_assignment = None  # default='warn'
pd.set_option('display.max_columns', 500)


train_df = pd.read_csv('/Users/apple/Desktop/Kaggle/train.csv')
test_df = pd.read_csv('/Users/apple/Desktop/Kaggle/test.csv')
#macro = pd.read_csv('Users/apple/Desktop/Kaggle/macro.csv')

train_df.shape

train_df.head()

#查看房价的整体趋势
plt.figure(figsize=(8,6))
plt.scatter(range(train_df.shape[0]),np.sort(train_df.price_doc.values))
plt.xlabel("index",fontsize=12)
plt.ylabel('price',fontsize=12)
plt.show(block=False)

plt.figure(figsize=(12,8))
sns.distplot(train_df.price_doc.values,bins=50,kde=True)
plt.show(block=False)

plt.figure(figsize=(12,8))
sns.distplot(np.log(train_df.price_doc.values), bins=50, kde=True)
plt.xlabel('price', fontsize=12)
plt.show(block=False)
#观察不同年份的房价中位值
train_df['yearmonth'] = train_df['timestamp'].apply(lambda x: x[:4]+x[5:7])
grouped_df = train_df.groupby('yearmonth')['price_doc'].aggregate(np.median).reset_index()

plt.figure(figsize=(12,8))
sns.barplot(grouped_df.yearmonth.values, grouped_df.price_doc.values, alpha=0.8, color="red")
plt.ylabel('Median Price', fontsize=12)
plt.xlabel('Year Month', fontsize=12)
plt.xticks(rotation='vertical')
plt.show(block=False)

#不同属性类别的数量
train_df=pd.read_csv("kaggle/russia/train.csv",parse_dates=['timestamp'])
dtype_df=train_df.dtypes.reset_index()
dtype_df.columns=["Count","ColumnsType"]
dtype_df.groupby("ColumnsType").aggregate("count").reset_index()

#不同列的缺失值数量
missing_df=train_df.isnull().sum(axis=0).reset_index()
missing_df.columns=['columns_name','missing_count']
missing_df=missing_df.ix[missing_df['missing_count']>0]
missing_df['missing_ratio']=missing_df['missing_count']/30471

ind=np.arange(missing_df.shape[0])
width=0.9
fig, ax = plt.subplots(figsize=(12,18))
rects = ax.barh(ind, missing_df.missing_count.values, color='y')
ax.set_yticks(ind)
ax.set_yticklabels(missing_df.columns_name.values, rotation='horizontal')
ax.set_xlabel("Count of missing values")
ax.set_title("Number of missing values in each column")
plt.show(block=False)

#做一个简单的xgboost模型，探索重要变量
for f in train_df.columns:
    if train_df[f].dtype=='object':
        lbl=preprocessing.LabelEncoder()
        lbl.fit(list(train_df[f].values))
        train_df[f]=lbl.transform(list(train_df[f].values))

train_y=train_df.price_doc.values
train_x=train_df.drop(['id','timestamp','price_doc'],axis=1)

xgb_params = {
    'eta': 0.05,
    'max_depth': 8,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}
dtrain = xgb.DMatrix(train_x, train_y, feature_names=train_x.columns.values)
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=100)

# plot the important features #
fig, ax = plt.subplots(figsize=(12,18))
xgb.plot_importance(model, height=0.8, ax=ax)
plt.show(block=False)

# 提取类似中位值，在百分之99.5处的价格
#这一步的目的在于，去掉过大和过小的点
ulimit = np.percentile(train_df.price_doc.values, 99.5)
llimit = np.percentile(train_df.price_doc.values, 0.5)
train_df['price_doc'].ix[train_df['price_doc']>ulimit] = ulimit
train_df['price_doc'].ix[train_df['price_doc']<llimit] = llimit

#观察重要变量和目标变量的分布情况
#总面积
col = "full_sq"
ulimit = np.percentile(train_df[col].values, 99.5)
llimit = np.percentile(train_df[col].values, 0.5)
train_df[col].ix[train_df[col]>ulimit] = ulimit
train_df[col].ix[train_df[col]<llimit] = llimit

plt.figure(figsize=(12,12))
sns.jointplot(x=np.log1p(train_df.full_sq.values), y=np.log1p(train_df.price_doc.values), size=10)
plt.ylabel('Log of Price', fontsize=12)
plt.xlabel('Log of Total area in square metre', fontsize=12)
plt.show(block=False)


#居住面积
col = "life_sq"
train_df[col].fillna(0, inplace=True)
ulimit = np.percentile(train_df[col].values, 95)
llimit = np.percentile(train_df[col].values, 5)
train_df[col].ix[train_df[col]>ulimit] = ulimit
train_df[col].ix[train_df[col]<llimit] = llimit

plt.figure(figsize=(12,12))
sns.jointplot(x=np.log1p(train_df.life_sq.values), y=np.log1p(train_df.price_doc.values),
              kind='kde', size=10)
plt.ylabel('Log of Price', fontsize=12)
plt.xlabel('Log of living area in square metre', fontsize=12)
plt.show(block=False)

#所在楼层
plt.figure(figsize=(12,8))
sns.countplot(x="floor", data=train_df)
plt.ylabel('Count', fontsize=12)
plt.xlabel('floor number', fontsize=12)
plt.xticks(rotation='vertical')
plt.show(block=False)

grouped_df=train_df.groupby("floor")['price_doc'].aggregate(np.median).reset_index()
plt.figure(figsize=(12,8))
sns.pointplot(grouped_df.floor.values, grouped_df.price_doc.values, alpha=0.8, color='blue')
plt.ylabel('Median Price', fontsize=12)
plt.xlabel('Floor number', fontsize=12)
plt.xticks(rotation='vertical')
plt.show(block=False)

#最大楼层
plt.figure(figsize=(12,8))
sns.countplot(x="max_floor", data=train_df)
plt.ylabel('Count', fontsize=12)
plt.xlabel('Max floor number', fontsize=12)
plt.xticks(rotation='vertical')
plt.show(block=False)

plt.figure(figsize=(12,8))
sns.boxplot(x="max_floor", y="price_doc", data=train_df)
plt.ylabel('Median Price', fontsize=12)
plt.xlabel('Max Floor number', fontsize=12)
plt.xticks(rotation='vertical')
plt.show(block=False)



























































































































































































































































































