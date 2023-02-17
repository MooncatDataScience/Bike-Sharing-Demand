import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# load
train = pd.read_csv('f:\\Ds/Bike Sharing Demand/train.csv')
test = pd.read_csv('f:\\Ds/Bike Sharing Demand/test.csv')

# 特徵工程
train['datetime'] = pd.to_datetime(train['datetime'])
train['year'] = train['datetime'].dt.year
train['month'] = train['datetime'].dt.month
train['hour'] = train['datetime'].dt.hour
train['dayofweek'] = train['datetime'].dt.dayofweek

test['datetime'] = pd.to_datetime(test['datetime'])
test['year'] = test['datetime'].dt.year
test['month'] = test['datetime'].dt.month
test['hour'] = test['datetime'].dt.hour
test['dayofweek'] = test['datetime'].dt.dayofweek

train = train.drop(['casual', 'registered'], axis=1)

categorical_features = ['season', 'holiday', 'workingday', 'weather']

for feature in categorical_features:
    train[feature] = train[feature].astype('category')
    test[feature] = test[feature].astype('category')

train = pd.get_dummies(train, columns=categorical_features) # train one hot
test = pd.get_dummies(test, columns=categorical_features) # test one hot

# 特徵縮放
from sklearn.preprocessing import StandardScaler

numeric_features = ['temp', 'atemp', 'humidity', 'windspeed']

scaler = StandardScaler()
train[numeric_features] = scaler.fit_transform(train[numeric_features])
test[numeric_features] = scaler.transform(test[numeric_features])

# 特徵交叉
train['year_month'] = (train['year'] - 2011)*12 + train['month']
test['year_month'] = (test['year'] - 2011) * 12 + test['month']


# 模型建立與訓練
train.set_index("datetime", inplace=True)
test.set_index("datetime", inplace=True)

# 預先處理與切割分組
X = train.drop(['count'], axis=1).values
y = train['count'].values
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

#轉成torch
X_train = torch.tensor(X_train, dtype=torch.float)
X_val = torch.tensor(X_val, dtype=torch.float)
y_train = torch.tensor(y_train, dtype=torch.float)
y_val = torch.tensor(y_val, dtype=torch.float)

# 建立線性回歸模型
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.output = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = torch.relu(self.hidden(x))
        x = self.output(x)
        return x.squeeze()

n_feature, n_hidden, n_output = len(X_train[0]), 32, 1

net = Net(n_feature, n_hidden, n_output)

# 優化器
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr = 0.01)

#訓練模型
for epoch in range(100):
    optimizer.zero_grad()
    y_pred = net(X_train)
    loss = criterion(y_pred, y_train)
    loss.backward()
    optimizer.step()

    # Print the loss every 10 epochs
    if (epoch+1) % 10 == 0:
        with torch.no_grad():
            y_val_pred = net(X_val)
            val_loss = criterion(y_val_pred, y_val)
            print(f'Epoch {epoch+1}, Training loss: {loss.item()}, Validation loss: {val_loss.item()}')


# 驗證模型
with torch.no_grad():
    y_train_pred = net(X_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    print(f'Train RMSE: {train_rmse}')

    y_val_pred = net(X_val)
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    print(f'Validation RMSE: {val_rmse}')



X_test = torch.tensor(test.values, dtype=torch.float)

with torch.no_grad():
    y_test_pred = net(X_test)


submission = pd.DataFrame({
    'datetime': test.index,
    'count': y_test_pred
})

submission.to_csv('submission.csv', index=False)