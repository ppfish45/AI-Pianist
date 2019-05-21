import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

import time
import copy

from ipywidgets import IntProgress
from IPython.display import display

import dataset

def load_data(spliter=['train', 'val'], color=['black'], difference=False, paddings=2):
    dataset.load_all_data(spliter, color, keypress=False, difference=difference, paddings=paddings)

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
        lstm_dropout=0.5,
        cnn=None
    ): 
        super(cnn_lstm, self).__init__()
        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim
        self.lstm = nn.LSTM(feature_dim, hidden_dim, num_layers=num_layers, dropout=lstm_dropout)
        self.hidden2logits = nn.Linear(hidden_dim, 1)
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
            Flatten()
        )
        if cnn:
            print('Loading pretrained cnn ...', end='')
            pretrained_dict = torch.load(cnn, map_location='cpu')
            model_dict = self.cnn.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.cnn.load_state_dict(model_dict)
            print(' Done')
            
    def forward(self, img_series):
        N = img_series.shape[0]
        cnn_out = self.cnn(img_series)
        tanh = nn.Tanh()
        lstm_out, _ = self.lstm(cnn_out.view(N, 1, -1))
        logits = self.hidden2logits(lstm_out[N - 1, 0, :])
        result = tanh(logits)
        return result

    def cnn_grad(self, grad):
        for param in self.cnn.parameters():
            param.requires_grad = grad

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class velocity_model():
    
    def __init__(self, height, width, lstm_dropout=0.0, cnn_model_path=None):
        self.epoch_total = 0
        fc_in = (width // 2 // 2) * (height // 2 // 2) * 32
        self.model = cnn_lstm(feature_dim=fc_in, hidden_dim=256, num_layers=2, cnn=cnn_model_path, lstm_dropout=lstm_dropout)
        if torch.cuda.is_available():
            self.model.to(device)

    def toggle_grad_cnn(self, grad=True):
        self.model.cnn_grad(grad)
    
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
        phase=['train', 'val'],
        color='black',
        learning_rate=1e-3,
        weight_lambda=0.0005,
        num_epoch=5,
        max_num=-1,
        best_path='model_best.tar',
        current_path='model_latest.tar',
        tsb_writer=None,
        tag='',
        decay_every=10,
        save_model=True
    ):
        model = self.model
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_lambda)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=decay_every, gamma=0.05)
        since = time.time()
        best_model = None
        best_loss = None

        best_path = time.strftime('[%Y%m%d]%H-%M-%S') + best_path
        current_path = time.strftime('[%Y%m%d]%H-%M-%S') + current_path

        print(f'The best model will be saved to {best_path} ...')
        print(f'Thhe latest model will be saved to {current_path} ...')

        for epoch in range(num_epoch):
            
            print('Epoch {}/{}'.format(epoch + 1, num_epoch), end='')
            self.epoch_total += 1
            _loss = dict()
            _diff = dict()

            for phase in phase:
                if phase == 'train':
                    scheduler.step()
                    model.train()
                else:
                    model.eval()
                
                running_loss = 0.0
                running_diff = 0.0
                total = dataset.get_lstm_data_num(phase, color) if max_num == -1 else max_num
                
                bar = IntProgress(max=total)
                display(bar)

                for i, (inputs, labels) in enumerate(dataset.lstm_data_batch(type=phase, color=color, max_num=total, need_bar=False)):

                    _labels = labels
                    inputs = torch.Tensor(inputs)
                    labels = torch.Tensor(np.array([labels]) / 63.5 - 1)
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    
                    optimizer.zero_grad()

                    outputs = model(inputs)
                    labels = torch.reshape(labels, [1])
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    running_loss += loss.item()
                    running_diff += np.abs((outputs.cpu().detach().numpy()[0] + 1) * 63.5 - _labels)

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    bar.value += 1
                    if i % 32 == 0:
                        bar.description = f'{bar.value} / {total}'
                
                bar.close()

                epoch_loss = running_loss / total
                epoch_diff = running_diff / total

                if epoch % 5 == 0:
                    print('{} Loss: {:.4f}, L1 Diff: {:.4f}'.format(
                        phase, epoch_loss, epoch_diff))
                
                _loss[phase] = epoch_loss
                _diff[phase] = epoch_diff

                if phase == 'val' and (best_loss == None or epoch_loss < best_loss):
                    best_loss = epoch_loss
                    best_model = copy.deepcopy(model.state_dict())
                    if save_model:
                        torch.save(model.state_dict(), best_path)
                
                if save_model:
                    torch.save(model.state_dict(), current_path)

            if tsb_writer and 'val' in phase and 'train' in phase:
                tsb_writer.add_scalars(f'{tag}/Loss', {'val': _loss['val'], 'train': _loss['train']}, self.epoch_total)
                tsb_writer.add_scalars(f'{tag}/L1 Diff', {'val': _diff['val'], 'train': _diff['train']}, self.epoch_total)


        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val loss: {:4f}'.format(best_loss))

        self.model = model