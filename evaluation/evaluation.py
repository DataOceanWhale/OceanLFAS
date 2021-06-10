import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import explained_variance_score, r2_score, mean_squared_error


def volatility(data):
    days = len(data)
    lnbal = [np.log(x) for x in data]
    diffbal = []
    for i in range(1, days):
        a = lnbal[i]-lnbal[i-1]
        diffbal.append(a)
    sigma = np.std(diffbal,ddof=1)
    return sigma


def get_score(pred, label, limit=None):
    exp_score = explained_variance_score(label, pred)
    r2 = r2_score(label, pred)
    rmse = np.sqrt(mean_squared_error(label, pred))
    mape = np.mean(abs(pred - label) / label)
    maxe = max(abs(pred - label) / label)

    model_metrics = {}
    model_metrics['exp_score'] = exp_score
    model_metrics['r2'] = r2
    model_metrics['rmse'] = rmse
    model_metrics['mape'] = mape
    model_metrics['maxe'] = maxe
    model_metrics['detailed_error'] = abs(pred - label) / label

    if limit is not None:
        n = np.sum((abs(pred - label) / label) >= limit)
        model_metrics['n'] = n

    return model_metrics


def model_plot(score, y_pred, y_true, y_pred_std=None):
    plt.figure()
    plt.plot(np.arange(len(y_pred)), y_pred, label='predict_value')
    if y_pred_std is not None:
        if len(y_pred_std.shape) == 1:
            plt.fill_between(np.arange(len(y_pred)), y_pred - 1.96 * y_pred_std,
                             y_pred + 1.96 * y_pred_std, alpha=0.5)
        elif len(y_pred_std.shape) == 2 and y_pred_std.shape[1] == 2:
            a_std = y_pred_std[:, 0]
            e_std = y_pred_std[:, 1]
            all_std = (a_std ** 2 + e_std ** 2) ** 0.5
            plt.fill_between(np.arange(len(y_pred)), y_pred - 1.96 * e_std,
                             y_pred + 1.96 * e_std, alpha=0.5, facecolor='blue')
            plt.fill_between(np.arange(len(y_pred)), y_pred - 1.96 * all_std,
                             y_pred + 1.96 * all_std, alpha=0.5, facecolor='lightblue')

    plt.plot(np.arange(len(y_true)), y_true, label='true value')
    plt.title('mape: {:.4f}, maxe: {:.4f}'.format(score['mape'], score['maxe']))

    plt.legend()
    plt.show()
