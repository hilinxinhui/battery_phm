import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import random

from statsmodels.tsa.ar_model import AutoReg # auto regression模型
from sklearn.svm import SVR # support vector machine模型
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim
import torch.nn.functional as F

# NASA PCoE B0005
# AR、SVR、MLP、LSTM、CNN

def build_sequences(data, time_steps):
    if type(data) != np.ndarray:
        data = np.array(data).reshape(-1, 1)
    return np.array([[j for j in data[i:i + time_steps]] for i in range(0, len(data) - time_steps + 1)])[:,:,0]

def seq_tar_gen(raw_data, tw=16, pw=1):
    sample = []
    for i in range(len(raw_data) - tw):
        sample.append([raw_data[i:i + tw], raw_data[i + tw: i + tw + pw]])
    return sample

def setup_seed(seed):
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现。
    torch.manual_seed(seed) # 为CPU设置随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed) # 为当前GPU设置随机种子
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU，为所有GPU设置随机种子
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

def sequence_target_generator(raw_data, window_size=16):
    sample = []
    for i in range(len(raw_data) - window_size):
        sample.append([raw_data[i:i + window_size], raw_data[i + window_size]])
    return sample

def generate_sequences(raw_data, tw, pw):
    data = dict()
    raw_data_length = len(raw_data)
    for i in range(raw_data_length - tw):
        sequence = raw_data[i:i + tw]
        target = raw_data[i + tw : i + tw + pw]
        data[i] = {
            "sequence": sequence,
            "target": target
        }
    return data

def feature_label_generator(raw_data, window_size=16):
    raw_data = np.array(raw_data) if type(raw_data) is not np.ndarray else raw_data
    # features, labels = [], []
    sample = []
    # for i in range(len(raw_data) - window_size):
    #     features.append(raw_data[i:i + window_size])
    #     labels.append(raw_data[i + window_size])
    # return features, labels
    for i in range(len(raw_data) - window_size):
        sample.append((raw_data[i:i + window_size], raw_data[i + window_size]))
    return sample

if __name__ == "__main__":

    # 数据集地址
    nasa_data_path = "../data/processed_data/nasa/nasa_capacity.npy"
    
    # 读取容量数据
    nasa_battery_name = "B0005"
    nasa_data = np.load(nasa_data_path, allow_pickle=True)
    nasa_pcoe_b0005 = np.load(nasa_data_path, allow_pickle=True)[0]


    # 准备数据集
    dataset = nasa_pcoe_b0005
    train_len = int(len(dataset) * 0.7)
    ar_train_dataset = dataset[0: train_len]
    ar_test_dataset = dataset[train_len: ] # 51个时间步

    # AR
    p = 16
    model = AutoReg(ar_train_dataset, lags=p)
    model_fit = model.fit()
    params = model_fit.params

    history = ar_train_dataset[-p:]
    history = np.hstack(history).tolist()
    ar_pred = []
    for t in range(len(ar_test_dataset)):
        y_hat = params[0]
        for i in range(p):
            y_hat += params[i + 1] * history[-1 - i]
        ar_pred.append(y_hat)
        history.append(ar_test_dataset[t])

    # SVR
    b05 = nasa_data[0].reshape(-1, 1)
    b06 = nasa_data[1].reshape(-1, 1)
    b07 = nasa_data[2].reshape(-1, 1)
    b18 = nasa_data[3].reshape(-1, 1)
    time_steps = 17 # 16个时间步输入，1个时间步输出
    b05 = build_sequences(b05, time_steps)
    b06 = build_sequences(b06, time_steps)
    b07 = build_sequences(b07, time_steps)
    b18 = build_sequences(b18, time_steps)
    dataset = (b05, b06, b07, b18)
    train_dataset = dataset[1: ]
    train_dataset = np.array([j for i in train_dataset for j in i])
    test_dataset = dataset[0]

    x_train, y_train = train_dataset[:,:time_steps - 1], train_dataset[:, [time_steps - 1]]
    x_test, y_test = test_dataset[:,:time_steps - 1], test_dataset[:, [time_steps - 1]]

    # 建模
    model = SVR(kernel="rbf", gamma=0.5, C=10, epsilon=0.05)
    # 预测
    model.fit(x_train, y_train[:,0])
    y_train_pred = model.predict(x_train).reshape(-1,1)
    y_test_pred = model.predict(x_test).reshape(-1,1)
    svr_pred = y_test_pred[-51: ]

    # MLP
    class MLPNet(nn.Module):
        def __init__(self, n_features=16, n_outputs=1, hidden_layers=[16, 8, 4, 2]):
            super(MLPNet, self).__init__()
            self.n_features = n_features
            self.n_outputs = n_outputs
            self.hidden_layers = hidden_layers
            self.n_mlp_layers = len(hidden_layers)

            self.input2mlp = nn.Linear(self.n_features, self.hidden_layers[0])
            mlp = []
            for i in range(self.n_mlp_layers):
                if i == self.n_mlp_layers - 1:
                    # mlp.append(nn.LeakyReLU())
                    mlp.append(nn.Linear(self.hidden_layers[-1], n_outputs))
                else:
                    # mlp.append(nn.LeakyReLU())
                    mlp.append(nn.Linear(self.hidden_layers[i], self.hidden_layers[i + 1]))
            self.mlp = nn.Sequential(*mlp)

        def forward(self, x):
            input_size = x.size(0) # input_size 即 batch_size
            x = self.input2mlp(x)
            x = self.mlp(x)
            return x
        
    class BatteryDataset(Dataset):
        def __init__(self, data):
            self.data = data
        def __len__(self):
            return len(self.data)
        def __getitem__(self, index):
            sample = self.data[index]
            seq = torch.FloatTensor(sample[0])
            tar = torch.FloatTensor(sample[1])
            return seq, tar

    b05 = nasa_data[0]
    b06 = nasa_data[1]
    b07 = nasa_data[2]
    b18 = nasa_data[3]

    window_size = 16
    b05 = seq_tar_gen(b05, window_size)
    b06 = seq_tar_gen(b06, window_size)
    b07 = seq_tar_gen(b07, window_size)
    b18 = seq_tar_gen(b18, window_size)

    USE_CUDA = torch.cuda.is_available()
    device = "cuda" if USE_CUDA else "cpu"

    epochs = 20
    lr = 1e-2
    window_size = 16
    batch_size = 16

    dataset = (b05, b06, b07, b18)

    train_dataset = dataset[1: ]
    train_dataset = np.array([j for i in train_dataset for j in i], dtype=object)
    test_dataset = dataset[0]

    train_dataset = BatteryDataset(train_dataset)
    test_dataset = BatteryDataset(test_dataset)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    model = MLPNet(hidden_layers=[16]).to(device)
    criterion = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses, val_losses = [], []

    train_loss, val_loss = 0.0, 0.0
    for epoch in range(epochs):
        model.train()
        
        # 训练
        for X, y in train_dataloader:
            optimizer.zero_grad()
            X, y = X.to(device), y.squeeze().to(device)
            preds = model(X).squeeze()
            loss = criterion(preds, y)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
        epoch_loss = train_loss / len(train_dataloader)
        train_losses.append(epoch_loss)

    y_pred, gt, y1, y2 = [], [], [], []
    cycle = [i + 1 for i in range(len(test_dataset))]
    for X, y in test_dataset:
        y = y.item()
        gt.append(y)
        y1.append(y * (1 + 0.05))
        y2.append(y * (1 - 0.05))
    model.eval()
    for cnt, (X, y) in enumerate(test_dataloader):
        X, y = X.to(device), y.squeeze().to(device)
        y_pred += model(X).cpu().squeeze().tolist()
    mlp_pred = y_pred[-51: ]

    # LSTM
    class LSTMNet(nn.Module):
        def __init__(self, n_features, n_hidden, n_outputs, sequence_len, n_lstm_layers=1, n_deep_layers=10, use_cuda=False, dropout=0.2):
            super(LSTMNet, self).__init__()
            
            # 初始化参数
            self.n_hidden = n_hidden
            self.n_lstm_layer = n_lstm_layers
            self.use_cuda = use_cuda

            # lstm层
            self.lstm = nn.LSTM(
                input_size=n_features,
                hidden_size=n_hidden,
                num_layers=n_lstm_layers,
                batch_first=True
            )
            
            # 第一个全连接层，使用dropout
            self.fc1 = nn.Linear(n_hidden * sequence_len, n_hidden)
            self.dropout = nn.Dropout(p=dropout)

            # 全连接层后接的dnn
            dnn_layers = []
            for i in range(n_deep_layers):
                if i == n_deep_layers - 1:
                    dnn_layers.append(nn.ReLU())
                    dnn_layers.append(nn.Linear(n_hidden, n_outputs))
                else:
                    dnn_layers.append(nn.ReLU())
                    dnn_layers.append(nn.Linear(n_hidden, n_hidden))
                    if dropout:
                        dnn_layers.append(nn.Dropout(p=dropout))
            self.dnn = nn.Sequential(*dnn_layers)

        def forward(self, x):
            # 初始化hidden state & cell state，并判断cuda可用状态
            hidden_sate = torch.zeros(self.n_lstm_layer, x.shape[0], self.n_hidden)
            cell_state = torch.zeros(self.n_lstm_layer, x.shape[0], self.n_hidden)
            if self.use_cuda:
                hidden_sate = hidden_sate.to(device)
                cell_state = cell_state.to(device)

            self.hidden = (hidden_sate, cell_state)

            x, h = self.lstm(x, self.hidden)
            x = self.dropout(x.contiguous().view(x.shape[0], -1))
            x = self.fc1(x)
            x = self.dnn(x)
            return x
    
    class BatteryDataset(Dataset):
        def __init__(self, data):
            self.data = data
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, index):
            sample = self.data[index]
            sequence = sample[0].reshape(-1, 1)
            target = sample[1]
            return torch.FloatTensor(sequence), torch.tensor([target])

    n_features = 1
    n_hidden = 100
    n_outputs = 1
    sequence_len = 16
    n_dnn_layers = 5

    lr = 1e-4
    epochs = 120

    b05 = nasa_data[0].astype(np.float32)
    b06 = nasa_data[1].astype(np.float32)
    b07 = nasa_data[2].astype(np.float32)
    b18 = nasa_data[3].astype(np.float32)

    window_size = 16
    b05 = sequence_target_generator(b05, window_size=window_size)
    b06 = sequence_target_generator(b06, window_size=window_size)
    b07 = sequence_target_generator(b07, window_size=window_size)
    b18 = sequence_target_generator(b18, window_size=window_size)

    dataset = [b05, b06, b07, b18]

    train_dataset = dataset[1: ]
    train_dataset = np.array([j for i in train_dataset for j in i], dtype=object)
    test_dataset = dataset[0]

    train_dataset = BatteryDataset(train_dataset)
    test_dataset = BatteryDataset(test_dataset)

    model = LSTMNet(
        n_features=n_features,
        n_hidden=n_hidden,
        n_outputs=n_outputs,
        sequence_len=sequence_len,
        n_deep_layers=n_dnn_layers,
        use_cuda=USE_CUDA
    ).to(device)
    criterion = nn.MSELoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    batch_size = 32
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size, shuffle=False)

    train_losses, val_losses = [], []

    epochs = 950
    for epoch in range(epochs):
        model.train()
        train_loss, val_loss = 0.0, 0.0
        
        # 训练
        for X, y in train_dataloader:
            optimizer.zero_grad()
            X, y = X.to(device), y.squeeze().to(device)
            preds = model(X).squeeze()
            loss = criterion(preds, y)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

    y_pred, gt = [], []
    cycle = [i + 1 for i in range(len(test_dataset))]
    for X, y in test_dataset:
        y = y.item()
        gt.append(y)
    model.eval()
    for cnt, (X, y) in enumerate(test_dataloader):
        X, y = X.to(device), y.squeeze().to(device)
        y_pred += model(X).cpu().squeeze().tolist()
    lstm_pred = y_pred[-51: ]

    # CNN
    class TSCNN(nn.Module):
        def __init__(self):
            super(TSCNN, self).__init__()
            self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=2)
            # self.relu = nn.LeakyReLU(negative_slope=0.1)
            self.relu = nn.ReLU()
            self.conv2 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=2)
            self.fc1 = nn.Linear(32 * 14, 50)
            self.fc2 = nn.Linear(50, 1)

        def forward(self, x):
            batch_size= x.size(0)
            x = self.conv1(x)
            x = self.relu(x)
            x = self.conv2(x)
            x = self.relu(x)
            x = x.view(batch_size, -1)
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            return x
        
    class BatteryDataset(Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, index):
            sequence = self.data[index][0].reshape(1, 16)
            target = self.data[index][1]
            return torch.FloatTensor(sequence), torch.FloatTensor([target])
        
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def train(model, train_dataloader, loss_fn, optimizer, epoch):
        model.train()
        for idx, (X, y) in enumerate(train_dataloader):
            X, y = X.to(device), y.to(device).reshape(-1, 1)
            pred = model(X)
            loss = loss_fn(y, pred)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    b05 = nasa_data[0]
    b06 = nasa_data[1]
    b07 = nasa_data[2]
    b18 = nasa_data[3]

    window_size = 16
    b05 = feature_label_generator(b05)
    b06 = feature_label_generator(b06)
    b07 = feature_label_generator(b07)
    b18 = feature_label_generator(b18)

    dataset = (b05, b06, b07, b18)

    train_dataset = dataset[1: ]
    train_dataset = np.array([j for i in train_dataset for j in i], dtype=object)
    test_dataset = dataset[0]

    model = TSCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00001)

    train_dataset = BatteryDataset(train_dataset)
    test_dataset = BatteryDataset(test_dataset)

    batch_size = 16
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)

    epochs = 500
    # print("训练开始")
    for epoch in range(epochs):
        train(model, train_dataloader, criterion, optimizer, epoch)
    # print("训练完成")

    y_pred, gt = [], []
    model.eval()
    for cnt, (X, y) in enumerate(test_dataset):
        X = X.to(device)
        y_pred.append(float(model(X)))
        gt.append(y)
    cnn_pred = y_pred[-51: ]

    # 作图
    plt.plot(figsize=(8, 6))
    plt.plot(ar_test_dataset)
    plt.plot(ar_pred)
    plt.plot(svr_pred)
    plt.plot(mlp_pred)
    plt.plot(lstm_pred)
    plt.plot(cnn_pred)
    plt.legend(["真值", "AR模型预测值", "SVR模型预测值", "MLP模型预测值", "LSTM模型预测值", "CNN模型预测值"])
    plt.xlabel("循环圈数")
    plt.ylabel("放电容量（Ah）")
    save_path = "../assets/thesis_figures/chapter_3/slide_figure_nasa.jpg"
    plt.savefig(save_path, dpi=1000, bbox_inches="tight")
    plt.tight_layout()
    plt.show()