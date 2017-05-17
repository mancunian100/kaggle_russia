import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mode
import time
from sklearn import model_selection, preprocessing
import xgboost as xgb
from sklearn.cross_validation import KFold
from xgboost.sklearn import XGBRegressor
import datetime
#now = datetime.datetime.now()
from sklearn.model_selection import train_test_split, GridSearchCV,cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score,roc_auc_score,mean_squared_error

pd.options.mode.chained_assignment = None  # default='warn'
pd.set_option('display.max_columns', 500)

train = pd.read_csv('kaggle/russia/train.csv')
test = pd.read_csv('kaggle/russia/test.csv')
macro = pd.read_csv('kaggle/russia/macro.csv')
id_test = test.id
full = pd.read_csv('kaggle/russia/full.csv')


#缺失值处理完毕，接下来挖掘新的特征
#房子本身的信息
full['floor_inverse']=full['max_floor']-full['floor']
full['floor_ratio']=full['floor']/full['max_floor']
#生活面积占总面积的比值,厨房面积占的比例，每个房间的面积大小
full['life_ratio']=full['life_sq']/full['full_sq']
full['kitch_ratio']=full['kitch_sq']/full['full_sq']
full['sq_per_room']=(full['life_sq']-full['kitch_sq'])/full['num_room']#0.31325




full['sq_per_room1']=full['full_sq']/full['num_room']#0.31543,结果下降


full['extra_area']=full['full_sq']-full['life_sq']#0.31343,结果微降

#人口信息
full['pop_density_raion']=full['raion_popul']/full['area_m']#0.31639,结果微降

full['young_proportion ']=full['young_all']/full['area_m']#0.31548退步明显
full['work_proportion ']=full['work_all']/full['area_m']#0.31443
full['retire_proportion ']=full['ekder_all']/full['area_m']
#已上三个都加上，结果是.031481
#售卖时距离建造的时间

#教育信息
full['ratio_preschool']=full['children_preschool']/full['preschool_quota']#0.31518
full['ratio_school']=full['children_school']/full['school_quota']
#加上以上两个特征结果0.31518


##=------------------------------------------------------------------
#特征处理完毕，接下来选择模型预测，融合
##=------------------------------------------------------------------


y_train = train["price_doc"]
x_train=full[:30471]
x_test=full[30471:]

#做本地的交叉验证
kf = KFold(x_train.shape[0], n_folds=5, random_state=1)
#cv借鉴了这个脚本https://www.kaggle.com/shaweji/script-v6/code
def rmse(y_true,y_pred):
    return np.sqrt(mean_squared_error(y_true,y_pred))
def rmse_cv(model, X_train, y):
    # RMSE with Cross Validation
    rmse= np.sqrt(-cross_val_score(model, X_train, y,
                                   scoring="neg_mean_squared_error", cv = kf))
    return(rmse)

model_xgb=xgb.XGBRegressor(
    learning_rate=0.05,
    max_depth=5,
    subsample=0.7,
    colsample_bytree=0.7,
    objective='reg:linear',
    silent=True,
    seed=2017
)

time_start=time.time()

# model_xgb.fit(x_train,y_train)


print("mean RMSE of XGBoost with CV: ", np.mean(rmse_cv(model_xgb, x_train, y_train)))
time_stop=time.time()
print(time_stop-time_start,"seconds")

model_xgb.fit(x_train,y_train)
y_predict=model_xgb.predict(x_test)
output = pd.DataFrame({'id': id_test, 'price_doc': y_predict})
output.to_csv('kaggle/russia/result.csv', index=False)







scoring="roc_auc"
X1, X2, y1, y2 = train_test_split(x_train, y_train, random_state=2017)

xgb_params = {
    'eta': 0.05,#0.05
    'max_depth': 5,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}

dtrain = xgb.DMatrix(X1, y1)
dtest = xgb.DMatrix(X2,y2)
watchlist = [(dtrain, 'train'), (dtest,'valid')]

model = xgb.train(xgb_params, dtrain, num_boost_round=1000, evals=watchlist, early_stopping_rounds=20,verbose_eval=50)

predictions=model.predict(xgb.DMatrix(X2))
fig, ax = plt.subplots(1, 1, figsize=(8, 13))
xgb.plot_importance(model,height=0.5, ax=ax)
plt.show(block=False)





#上面是本地的交叉验证，下面是预测得到结果,其实是一样的，cv就是自带的交叉验证
xgb_params = {
    'eta': 0.05,#0.05
    'max_depth': 5,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}

dtrain = xgb.DMatrix(x_train, y_train)
dtest = xgb.DMatrix(x_test)

cv_output = xgb.cv(xgb_params, dtrain, num_boost_round=1000, early_stopping_rounds=20,verbose_eval=50, show_stdv=False)
cv_output[['train-rmse-mean', 'test-rmse-mean']].plot()

num_boost_rounds = len(cv_output)
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round= num_boost_rounds)

fig, ax = plt.subplots(1, 1, figsize=(8, 13))
xgb.plot_importance(model,  height=0.5, ax=ax)
plt.show(block=False)

y_predict = model.predict(dtest)
output = pd.DataFrame({'id': id_test, 'price_doc': y_predict})
output.head()


output.to_csv('kaggle/russia/result.csv', index=False)













#筛选出一些重要性较高特征
features=[]
a=model.get_fscore()
importance_map=sorted(a.items(), key=lambda d:d[1],reverse=True)
for x in importance_map[:220]:
    # print(x)
    features.append(x[0])



y_train = train["price_doc"]
x_train=full[:30471][features]
x_test=full[30471:][features]
xgb_params = {
    'eta': 0.05,#学习率
    'max_depth': 5,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}

dtrain = xgb.DMatrix(x_train, y_train)
dtest = xgb.DMatrix(x_test)

cv_output = xgb.cv(xgb_params, dtrain, num_boost_round=1000, early_stopping_rounds=20,verbose_eval=50, show_stdv=False)
cv_output[['train-rmse-mean', 'test-rmse-mean']].plot()

num_boost_rounds = len(cv_output)
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round= num_boost_rounds)

fig, ax = plt.subplots(1, 1, figsize=(8, 13))
xgb.plot_importance(model,  height=0.5, ax=ax)
plt.show(block=False)




y_predict = model.predict(dtest)
output = pd.DataFrame({'id': id_test, 'price_doc': y_predict})
output.head()


output.to_csv('kaggle/russia/result.csv', index=False)




















































































































































