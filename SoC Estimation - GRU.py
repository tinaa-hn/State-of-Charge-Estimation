# %% libraries
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
import matplotlib.pyplot as plt

#%% Dataset generation

num_steps = 86400  #total number of time steps for 24 hours
voltage_mean = 1.5  #average voltage
voltage_std = 0.1  #voltage sd
current_mean = 200  #average current in A
current_std = 1 #current standard deviation
concentration_interval = 900  #concentration measurement interval (every 15 mins)
initial_concentration = 2
M_max = 2
M_min = 0.4
initial_soc = 0
battery_capacity_Ah = 4800  #battery capacity in Ah

np.random.seed(42)
voltage = np.random.normal(voltage_mean, voltage_std, num_steps)
current = np.random.normal(current_mean, current_std, num_steps)

#initializing concentration and SoC arrays
concentration = np.full(num_steps, np.nan)
SoC = np.full(num_steps, np.nan)  
concentration[0] = initial_concentration
SoC[0] = initial_soc


#using coulomb counting method to update SoC and concentration every 900 seconds
for i in range(concentration_interval, num_steps, concentration_interval):
    print (i)
    #ensuring we don't go out of bounds
    if i >= num_steps:
        break

    #calculating the total charge over the last interval
    total_charge = np.sum(current[i - concentration_interval:i]) * (1 / 3600)  # Convert to hours
    delta_soc = total_charge / battery_capacity_Ah
    #print (delta_soc)
    
    #updating SoC
    SoC[i] = SoC[i - concentration_interval] + delta_soc
    #print (SoC[i])
    #soc bounds
    SoC[i] = max(0, min(SoC[i], 1))
    
    #updating concentration based on SoC
    concentration[i] = M_max - SoC[i] * (M_max - M_min)

# Forward fill SoC values for plotting
#SoC = pd.Series(SoC).ffill().values

data = pd.DataFrame({
    'time': np.arange(num_steps),
    'voltage': voltage,
    'current': current,
    'concentration': concentration,
    'SoC': SoC
})

data.to_csv('simulated_battery_data_corrected.csv', index=False)

#%% Preprocessing

#load and preprocess the dataset
data = pd.read_csv('simulated_battery_data_corrected.csv')


#handling the  missing values
data['SoC'].fillna(data['SoC'].mean(), inplace=True)
data.dropna(subset=['voltage', 'current', 'concentration'], inplace=True)


features = data.drop('SoC', axis=1).values
target = data['SoC'].values

scaler = StandardScaler()
features = scaler.fit_transform(features)




features_tensor = torch.tensor(features, dtype=torch.float32)
target_tensor = torch.tensor(target, dtype=torch.float32)


dataset = TensorDataset(features_tensor, target_tensor)

# GRU model with dropout
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.3):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h_0)
        out = self.fc(self.dropout(out[:, -1, :]))
        return out


#%%

#hyperparam tuning setup
learning_rates = [0.001, 0.0005]
batch_sizes = [32, 64]
hidden_sizes = [64, 128]
num_layers_list = [2, 3]
dropout_rates = [0.3, 0.4]
weight_decays = [0.0001, 0.0005]
best_mae = float('inf')
best_params = {}

kf = KFold(n_splits=3, shuffle=True, random_state=42)

for lr in learning_rates:
    for batch_size in batch_sizes:
        for hidden_size in hidden_sizes:
            for num_layers in num_layers_list:
                for dropout in dropout_rates:
                    for weight_decay in weight_decays:
                        fold_maes = []

                        for train_index, val_index in kf.split(features_tensor):
                            X_train, X_val = features_tensor[train_index], features_tensor[val_index]
                            y_train, y_val = target_tensor[train_index], target_tensor[val_index]

                            #reshape for GRU: (batch_size, sequence_length, input_size)
                            X_train = X_train.view(-1, 1, X_train.shape[1])
                            X_val = X_val.view(-1, 1, X_val.shape[1])

                            train_dataset = TensorDataset(X_train, y_train)
                            val_dataset = TensorDataset(X_val, y_val)

                            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

                            input_size = X_train.shape[2]
                            model = GRUModel(input_size, hidden_size, num_layers, dropout)
                            criterion = nn.MSELoss()
                            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
                            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

                            #early-stopping params
                            early_stopping_patience = 20
                            early_stopping_counter = 0
                            best_val_loss = float('inf')

                            num_epochs = 50

                            for epoch in range(num_epochs):
                                model.train()
                                running_loss = 0.0
                                for X_batch, y_batch in train_loader:
                                    optimizer.zero_grad()
                                    outputs = model(X_batch).squeeze()
                                    loss = criterion(outputs, y_batch)
                                    loss.backward()
                                    optimizer.step()
                                    running_loss += loss.item() * X_batch.size(0)

                                train_loss = running_loss / len(train_loader.dataset)

                                model.eval()
                                with torch.no_grad():
                                    val_loss = sum(criterion(model(X_batch).squeeze(), y_batch).item() for X_batch, y_batch in val_loader) / len(val_loader.dataset)

                                scheduler.step()

                                if val_loss < best_val_loss:
                                    best_val_loss = val_loss
                                    early_stopping_counter = 0
                                else:
                                    early_stopping_counter += 1

                                if early_stopping_counter >= early_stopping_patience:
                                    break

                            #evaluating the model on the validation set
                            model.eval()
                            with torch.no_grad():
                                y_pred = model(X_val).squeeze().numpy()
                                val_mae = mean_absolute_error(y_val.numpy(), y_pred)
                                fold_maes.append(val_mae)

                        avg_mae = np.mean(fold_maes)
                        if avg_mae < best_mae:
                            best_mae = avg_mae
                            best_params = {'learning_rate': lr, 'batch_size': batch_size, 'hidden_size': hidden_size, 'num_layers': num_layers, 'dropout': dropout, 'weight_decay': weight_decay}

print(f"Best Hyperparameters: {best_params}")
print(f"Best Validation MAE: {best_mae:.4f}")


#%% Training

#training the final model with best hyperparams
X_train, X_test, y_train, y_test = train_test_split(features_tensor, target_tensor, test_size=0.2, random_state=42)



X_train = X_train.view(-1, 1, X_train.shape[1])
X_test = X_test.view(-1, 1, X_test.shape[1])

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=best_params['batch_size'], shuffle=False)

input_size = X_train.shape[2]
hidden_size = best_params['hidden_size']
num_layers = best_params['num_layers']
dropout = best_params['dropout']
weight_decay = best_params['weight_decay']
model = GRUModel(input_size, hidden_size, num_layers, dropout)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=best_params['learning_rate'], weight_decay=weight_decay)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

early_stopping_patience = 20
early_stopping_counter = 0
best_val_loss = float('inf')

num_epochs = 200
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch).squeeze()
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * X_batch.size(0)

    train_loss = running_loss / len(train_loader.dataset)
    train_losses.append(train_loss)

    model.eval()
    with torch.no_grad():
        val_loss = sum(criterion(model(X_batch).squeeze(), y_batch).item() for X_batch, y_batch in test_loader) / len(test_loader.dataset)
        val_losses.append(val_loss)

    scheduler.step()

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stopping_counter = 0
    else:
        early_stopping_counter += 1

    if early_stopping_counter >= early_stopping_patience:
        print(f"Early stopping at epoch {epoch+1}")
        break

    if epoch % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')


#%%Plotting and evaluation on test set


plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#evaluating the model on the test set
model.eval()
with torch.no_grad():
    y_pred = model(X_test).squeeze().numpy()
    test_mae = mean_absolute_error(y_test.numpy(), y_pred)
    test_r2 = r2_score(y_test.numpy(), y_pred)
    print(f'Test MAE: {test_mae:.4f}')
    print(f'Test R-squared: {test_r2:.4f}')
