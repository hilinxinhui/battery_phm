import os
import scipy
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

# 从 mat 文件读取循环数据
def load_mat(file_path):
    filename = os.path.basename(file_path).split(".")[0] # eg.B0005
    data = scipy.io.loadmat(file_path)
    # print(type(data)) # dict
    # print(data.keys()) # dict_keys(['__header__', '__version__', '__globals__', 'B0005'])
    # print(data["B0005"][0][0][0][0].shape) # (616,)，实验持续了616个循环
    # print(type(data["B0005"][0][0][0][0][0])) # 用scipy读取mat文件，读取到结构体返回 numpy.void 类型数据

    cols = data[filename][0][0][0][0]
    
    data = []
    
    for col in cols:
        _ = {}
        parameter = {}
        keys = list(col[3][0][0].dtype.fields.keys())
        if str(col[0][0]) != "impedance":
            for idx, key in enumerate(keys):
                parameter[key] = list(col[3][0][0][idx][0])

        operation_type = str(col[0][0])
        temperature = str(col[1][0][0])
        time = str(datetime(
            int(col[2][0][0]), int(col[2][0][1]), int(col[2][0][2]), int(col[2][0][3]), int(col[2][0][4]), int(col[2][0][5])
        ))
        _["type"], _["temperature"], _["time"], _["data"] = operation_type, temperature, time, parameter
        data.append(_)

    return data

battery_names = ["B0005", "B0006", "B0007", "B0018"]
battery_init_capacity = [1.86, 2.04, 1.89, 1.86]
data_root_path = "../data/raw_data/nasa/"
data_path = [os.path.join(data_root_path, name + ".mat") for name in battery_names]

nasa_data = {}
for name, path in zip(battery_names, data_path):
    nasa_data[name] = load_mat(path)
keys = list(nasa_data.keys())
for key in keys:
    print(nasa_data[key][0]["data"].keys())