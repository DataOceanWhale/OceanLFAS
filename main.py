from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import pandas as pd
import numpy as np

from utils.utils import rolling_fit, gen_time_decay_seq
from evaluation.evaluation import get_score, model_plot

rolling = False
window = 360  # moving window
subtract = True # 学习对象是否为差值
standardize = True
default_strategy = None  # 'final_30_20'
day = 30 #预测T+n天后的balance
method = 'bnn' # 'xbg' or 'bnn' or 'baseline'
have_std = False
support_time_decay = False

pd_x_train = pd.read_csv('') # tain_dataset features
pd_x_test = pd.read_csv('') # test_dataset features
y_train = pd.read_csv('') # tain_dataset label
y_test = pd.read_csv('') # test_dataset label
x_train = pd_x_train.values.astype(np.float)
x_test = pd_x_test.values.astype(np.float)
y_prev_train = pd_x_train[''].to_numpy() # train集里n天前的balance，用于计算balance差值
y_prev_test = pd_x_test[''].to_numpy() # test集里n天前的balance，用于计算balance差值

def model_fit(model, x_train, x_test, y_train, have_std=False, time_decay_strategy=default_strategy):
    x_train, y_train = x_train[:-day], y_train[:-day]
    if support_time_decay and time_decay_strategy is not None:
        time_decay = gen_time_decay_seq(time_decay_strategy, x_train.shape[0])
        model.fit(x_train, y_train, time_decay=time_decay)
    else:
        model.fit(x_train, y_train)
    if not have_std:
        y_pred_train = model.predict(x_train)
        y_pred_test = model.predict(x_test)
        return y_pred_test, None, y_pred_train, None
    else:
        y_pred_train, y_std_train = model.predict(x_train, return_std=True)
        y_pred_test, y_std_test = model.predict(x_test, return_std=True)
        return y_pred_test, y_std_test, y_pred_train, y_std_train


if method == 'xgb':
    from xgboost import XGBRegressor
    reg = XGBRegressor()
elif method == 'bnn':
    from algorithms.bnn_regressor import BNNRegressor
    reg = BNNRegressor(x_dim=x_train.shape[1], hidden_layer_sizes=[50,50,50], optimizer='sghmc', learning_rate=1e-3, epochs=100, verbose=True, n_particles=20, update_logstd=False)
    have_std = True
    support_time_decay = True
elif method == 'baseline':
    from algorithms.baseline_regressor import BaselineRegressor
    reg = BaselineRegressor()
    subtract = True
    rolling = False


if subtract:
    y_train -= y_prev_train
    y_test -= y_prev_test

if standardize:
    from sklearn.preprocessing import StandardScaler
    scaler_x = StandardScaler()
    n_features = x_train.shape[-1]
    x_train = scaler_x.fit_transform(x_train)
    x_test = scaler_x.transform(x_test)
    y_mean, y_std = np.mean(y_train), np.std(y_train)
    y_train = (y_train - y_mean) / y_std
    y_test = (y_test - y_mean) / y_std

if not rolling:
    y_pred_test, y_std_test, y_pred_train, y_std_train = model_fit(
        reg, x_train, x_test, y_train, have_std=have_std)
else:
    y_pred_test, y_std_test, y_pred_train, y_std_train = rolling_fit(
        lambda x_train, x_test, y_train: model_fit(reg, x_train, x_test, y_train, have_std=have_std),
        x_train, x_test, y_train, y_test, window=window
    )

if standardize:
    y_pred_test = y_pred_test * y_std + y_mean
    y_pred_train = y_pred_train * y_std + y_mean
    if y_std_test is not None:
        y_std_test *= y_std
        y_std_train *= y_std
    y_train = y_train * y_std + y_mean
    y_test = y_test * y_std + y_mean

if subtract:
    y_train += y_prev_train
    y_test += y_prev_test
    y_pred_train += y_prev_train
    y_pred_test += y_prev_test

score_train = get_score(y_pred_train, y_train)
model_plot(score_train, y_pred_train, y_train, y_std_train)
if subtract:
    model_plot(score_train, y_pred_train-y_prev_train,
               y_train-y_prev_train, y_std_train)

score_test = get_score(y_pred_test, y_test)
model_plot(score_test, y_pred_test, y_test, y_std_test)
if subtract:
    model_plot(score_test, y_pred_test-y_prev_test,
               y_test-y_prev_test, y_std_test)
