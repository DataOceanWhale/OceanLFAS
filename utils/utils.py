import numpy as np
import tqdm

def gen_time_decay_seq(time_decay_strategy, length):
    if time_decay_strategy == 'linear':
        time_decay = np.linspace(0, 1, length)
    elif time_decay_strategy[:5] == 'final':
        _, days, intensity = time_decay_strategy.split('_')
        days = int(days)
        intensity = float(intensity)
        time_decay = np.ones([length])
        time_decay[-days:] = np.linspace(1, intensity, days)
    time_decay /= np.mean(time_decay)

def rolling_fit(fit_func, x_train, x_test, y_train, y_test, window=None):
    x_past = x_train
    y_past = y_train
    y_pred_test = []
    y_std_test = []
    _, _, y_pred_train, y_std_train = fit_func(x_train, x_test, y_train)
    with tqdm.trange(x_test.shape[0]) as trg:
        for i in trg:
            x_sing = x_test[[i]]
            y_sing = y_test[[i]]
            if window is not None:
                x_past_window, y_past_window = x_past[-window:], y_past[-window:]
            y_pred_sing, y_std_sing, _, _ = fit_func(x_past_window, x_sing, y_past_window)
            y_pred_test.append(y_pred_sing[0])
            if y_std_sing is not None:
                y_std_test.append(y_std_sing[0])
            x_past = np.concatenate([x_past, x_sing])
            y_past = np.concatenate([y_past, y_sing])
    y_pred_test = np.array(y_pred_test)
    if y_std_test == []:
        y_std_test = None
    else:
        y_std_test = np.array(y_std_test)
    return y_pred_test, y_std_test, y_pred_train, y_std_train
