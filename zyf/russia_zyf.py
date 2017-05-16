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
id_test = test.id

##=------------------------------------------------------------------
#特征处理完毕，接下来选择模型预测，融合
##=------------------------------------------------------------------

full = pd.read_csv('kaggle/russia/full1.csv')
y_train = train["price_doc"]
x_train=full[:30471]
x_test=full[30471:]
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



























































































































































