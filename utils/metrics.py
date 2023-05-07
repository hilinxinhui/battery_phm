from math import sqrt
import numpy as np
from sklearn.metrics import max_error, mean_absolute_error, mean_squared_error


def evaluation(y_true, y_predict):
    """返回MaxE、MAE和RMSE"""
    y_true, y_predict = np.array(y_true).flatten(), np.array(y_predict).flatten()
    maxe = max_error(y_true, y_predict)
    mae = mean_absolute_error(y_true, y_predict)
    mse = mean_squared_error(y_true, y_predict)
    rmse = sqrt(mse)
    return maxe, mae, rmse
    

def relative_error(y_true, y_predict, threshold):
    """依据SOH预测RUL并计算相对误差"""
    rul_true, rul_pred = 0, 0
    for i in range(len(y_true) - 1):
        if threshold >= y_true[i] and threshold >= y_true[i + 1]:
            rul_true = i - 1
            break

    for i in range(len(y_predict)):
        if threshold >= y_predict[i]:
            rul_pred = i - 1
            break
    return abs(rul_true - rul_pred) / rul_true