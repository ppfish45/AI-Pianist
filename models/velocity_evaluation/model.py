import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import time
import copy

from ipywidgets import IntProgress
from IPython.display import display

class Flatten(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)

class cnn_lstm(nn.Module):

    def __init__(
        self,
        feature_dim,
        hidden_dim,
        num_layers,
        cnn=None
    ):
        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim
        self.lstm = nn.LSTM(feature_dim, hidden_dim, num_layers=num_layers)
        self.hidden2logits = nn.Linear(hidden_dim, 1)
        if cnn == None:
            self.cnn = torch.nn.Sequential(
                torch.nn.Conv2d(3, 16, 3, padding=1),
                torch.nn.ReLU(),
                torch.nn.BatchNorm2d(16),
                torch.nn.Conv2d(16, 16, 3, padding=1),
                torch.nn.ReLU(),
                torch.nn.BatchNorm2d(16),
                torch.nn.MaxPool2d(2),
                torch.nn.Conv2d(16, 32, 3, padding=1),
                torch.nn.ReLU(),
                torch.nn.BatchNorm2d(32),
                torch.nn.Conv2d(32, 32, 3, padding=1),
                torch.nn.ReLU(),
                torch.nn.BatchNorm2d(32),
                torch.nn.MaxPool2d(2),
                Flatten(),
            )
        else:
            self.cnn = cnn
            
    def forward(self, img_series):
        N = img_series[0]
        cnn_out = self.cnn(img_series)
        lstm_out, _ = self.lstm(cnn_out.view(N, 1, -1))
        logits = self.hidden2logits(lstm_out[1, 0, :])
        result = (nn.Tanh(logits) + 1) * 127.5
        return result

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class velocity_model():
    
    def __init__(self, height, width):
        fc_in = (width // 2 // 2) * (height // 2 // 2) * 32
        self.model = cnn_lstm(feature_dim=fc_in, hidden_dim=256, num_layers=2)
        if torch.cuda.is_available():
            self.model.to(device)
    
    def load_model(self, path):
        print(f'Trying to load model from {path} ...', end='')
        self.model.load_state_dict(torch.load(path, map_location='cpu'))
        if torch.cuda.is_available():
            self.model.to(device)
        print(' Done')

    def evaluate(self, X):
        with torch.no_grad():
            self.model.eval()
            inputs = torch.Tensor(X)
            inputs = inputs.to(device)
            outputs = torch.squeeze(self.model(inputs))
            outputs = torch.reshape(outputs, [-1, 1])
            out = outputs.cpu().numpy()
        return out

    def train(
        self,
        learning_rate=1e-3,
        num_epoch=5,
        max_num=-1,
        best_path='model_best.tar',
        current_path='model_latest.tar',
        decay_every=10,
        save_model=True
    ):
        model = self.model
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=decay_every, gamma=0.05)
        since = time.time()
        best_model = None
        best_loss = None

        for epoch in range(num_epoch):
            print('Epoch {}/{}'.format(epoch + 1, num_epoch))
            for phase in ['train', 'val']:
                if phase == 'train':
                    scheduler.step()
                    model.train()
                else:
                    model.eval()
                
                running_loss = 0.0
                total = dataset.get_num_of_data(phase) if max_num == -1 else max_num
                
                bar = IntProgress(max=total)
                display(bar)

                for inputs, labels in dataset.lstm_ (type=phase, max_num=total):
                    
                    inputs = torch.Tensor(inputs)
                    labels = torch.Tensor(labels)
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    
                    optimizer.zero_grad()

                    outputs = model(inputs)
                    labels = torch.reshape(labels, [-1, 1])
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    running_loss += loss.item()

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    bar.value += 1
                    bar.description = f'{bar.value} / {total}'
                
                bar.close()

                epoch_loss = running_loss / total

                print('{} Loss: {:.4f}'.format(
                    phase, epoch_loss))

                if phase == 'val' and (best_loss == None or epoch_loss < best_loss):
                    best_loss = epoch_loss
                    best_model = copy.deepcopy(model.state_dict())
                    torch.save(model.state_dict(), best_path)
                    print(f'The best model has been saved to {best_path} ...')

                torch.save(model.state_dict(), current_path)
                print(f'Current mode has been saved to {current_path} ...')

                print()

            time_elapsed = time.time() - since
            print('Training complete in {:.0f}m {:.0f}s'.format(
                time_elapsed // 60, time_elapsed % 60))
            print('Best val loss: {:4f}'.format(best_loss))
