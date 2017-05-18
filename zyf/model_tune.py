import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mode

from sklearn import model_selection, preprocessing
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
import datetime
#now = datetime.datetime.now()
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score,roc_auc_score

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
'''
###-----------------------------------------------------------------
###下面是对结果没有提升的特征
###----------------------------------------------------
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
full['ratio_preschool']=full['children_preschool']/full['preschool_quota']
full['ratio_preschool']=full['children_preschool']/full['preschool_quota']
#加上以上两个特征结果0.31518
'''

##=------------------------------------------------------------------
#特征处理完毕，接下来选择模型预测，融合
##=------------------------------------------------------------------

y_train = train["price_doc"]
x_train=full[:30471]
x_test=full[30471:]

def modelfit(alg, x_train, y_train ,useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(x_train.values, label=y_train.values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                          metrics='rmse', early_stopping_rounds=early_stopping_rounds, show_stdv=False)
        alg.set_params(n_estimators=cvresult.shape[0])

    #Fit the algorithm on the data
    alg.fit(x_train, y_train,eval_metric='rmse')

    #Predict training set:
    # dtrain_predictions = alg.predict(x_train)
    dtrain_predprob = alg.predict_proba(x_train)[:,1]

    #Print model report:
    print("\nModel Report")
    # print("Accuracy:{}".format(accuracy_score(y_train.values,dtrain_predictions)))
    # print("AUC Score (Train):{}".format(roc_auc_score(y_train,dtrain_predprob)))
    feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    plt.show(block=False)


#不管任何参数，都用默认饿的，观察一下结果
xgb_model=XGBClassifier(objective='reg:linear')
modelfit(xgb_model,x_train,y_train)
xgb_model=XGBClassifier(learning_rate =0.1,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'reg:linear',
 nthread=4,
 scale_pos_weight=1,
 seed=2017)
modelfit(xgb_model,x_train,y_train)



#第一步，确定学习速率，和tree_based参数调优的估计器数量










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
#特征重要性画图
fig, ax = plt.subplots(1, 1, figsize=(8, 13))
xgb.plot_importance(model,  height=0.5, ax=ax)
plt.show(block=False)




y_predict = model.predict(dtest)
output = pd.DataFrame({'id': id_test, 'price_doc': y_predict})
output.head()


output.to_csv('kaggle/russia/result.csv', index=False)































































#筛选出一些重要性较高特征，再进行预测
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
