import torch
import torch.nn as nn
import torch.optim as optim

import time
import copy
import tqdm
import cv2
import numpy as np

from ipywidgets import IntProgress
from IPython.display import display

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
black_mask = np.array(
    [1, 4, 6, 9, 11, 13, 16, 18, 21, 23, 25, 28, 30, 33, 35, 37, 40, 42, 45, 47, 49, 52, 54, 57, 59, 61, 64,
     66, 69, 71, 73, 76, 78, 81, 83, 85])
white_mask = np.array(
    [0, 2, 3, 5, 7, 8, 10, 12, 14, 15, 17, 19, 20, 22, 24, 26, 27, 31, 32, 34, 36, 38, 39, 41, 43, 44, 46, 48,
     50, 51, 53, 55, 56, 58, 60, 62, 63, 65, 67, 68, 70, 72, 74, 75, 77, 79, 80, 82, 84, 86, 87])


class ModelWrapper():
    def __init__(self, model, loss_fn, optim=optim.Adam):
        """
        model: A model object, usually some nn.Sequential(...) variable
        loss_fn: A loss function *class*. 
        optim: A optim *class*. Default is torch.optim.Adam
        Example Usage: model = ModelWrapper(torch.nn.Sequential([torch.nn.Linear(10, 1)]), torch.nn.MCELoss, torch.optim.Adam)
        """
        self.model = model
        self.loss_fn = loss_fn
        self.optim = optim

        if torch.cuda.is_available():
            self.model.to(device)

    def load_model(self, path):
        print(f'Trying to load model from {path} ...')
        self.model.load_state_dict(torch.load(path, map_location='cpu'))
        if torch.cuda.is_available():
            self.model.to(device)
        print('done')

    def evaluate(self, X):
        '''
        please use NCHW format
        '''
        self.model.eval()
        with torch.no_grad():
            # inputs = torch.Tensor(X)
            X.to(device)
            outputs = self.model(X)
            return outputs

    def get_accuracy(self, X, y, threshold=0.5):
        y_pred = self.evaluate(X).cpu()
        y = y.cpu()
        sm_mask = y_pred < threshold
        lg_mask = y_pred >= threshold
        y_pred[sm_mask] = 0
        y_pred[lg_mask] = 1
        if y.shape[1] == 88:
            white_acc = [[None, None], [None, None]]
            black_acc = [[None, None], [None, None]]
            white_y = y[:, white_mask]
            black_y = y[:, black_mask]
            white_y_pred = y_pred[:, white_mask]
            black_y_pred = y_pred[:, black_mask]
            for i in (0, 1):
                for j in (0, 1):
                    white_acc[i][j] = torch.sum((white_y_pred == i) & (white_y == j)).tolist()
                    black_acc[i][j] = torch.sum((black_y_pred == i) & (black_y == j)).tolist()
            return white_acc, black_acc

        else:
            acc = [[None, None], [None, None]]
            for i in (0, 1):
                for j in (0, 1):
                    acc[i][j] = torch.sum((y_pred == i) & (y == j)).tolist()
            return acc

    def train(
            self,
            dataset,
            key_type=None,
            batch_size=64,
            learning_rate=1e-3,
            num_epochs=5,
            max_num=-1,
            best_path='keyboard_model_best.tar',
            current_path='keyboard_model_latest.tar',
            decay_every=10,
            save_model=True,
            method=2
    ):
        """
        dataset must have these APIs: 
        get_num_of_data(phase: "train"|"val") -> int, 
        data_batch(type: "train"|"val", batch_size: int, max_num=max_num_for_this_epoch: int, method: int) -> iterable
        """
        model = self.model
        criterion = self.loss_fn()
        optimizer = self.optim(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=decay_every, gamma=0.05)

        since = time.time()

        best_model_wts = copy.deepcopy(model.state_dict())
        best_loss = None

        print("Accuracy Matrix: both_0, pred_0_but_out_1, pred_1_but_out_0, both_1")

        for epoch in range(num_epochs):

            print('Epoch {}/{}'.format(epoch + 1, num_epochs))

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    scheduler.step()
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                # Iterate over data.

                max_num_for_this_epoch = max_num if phase == 'train' else -1

                total = dataset.get_num_of_data(phase) if max_num == -1 else max_num

                bar = IntProgress(max=total)

                display(bar)

                for i in dataset.data_batch(type=phase, batch_size=batch_size, max_num=max_num_for_this_epoch,
                                            method=method):
                    if method == 2:
                        inputs, labels = i
                    else:
                        if key_type == 'white':
                            inputs = i[0]
                            labels = i[2]
                        elif key_type == 'black':
                            inputs = i[1]
                            labels = i[3]
                        else:
                            raise ValueError("Fuck you. Specify key type 'white'|'black'")

                    inputs = torch.Tensor(inputs)
                    labels = torch.Tensor(labels)
                    if torch.cuda.is_available():
                        inputs = inputs.cuda()
                        labels = labels.cuda()
                    # inputs.to(device)
                    # labels.to(device)

                    # print(inputs.shape)  # inputs[0] : CHW
                    # print(labels.shape)
                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    outputs = model(inputs)
                    # if method == 2:
                    #     labels = torch.reshape(labels, [-1, 88])
                    # else:
                    #     labels = torch.reshape(labels, [-1, 1])
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    # statistics
                    running_loss += loss.item() * batch_size

                    # free unoccupied memory
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    bar.value += batch_size
                    bar.description = f'{bar.value} / {total}'

                bar.close()

                epoch_loss = running_loss / dataset.get_num_of_data(phase)

                print('{} Loss: {:.4f}'.format(
                    phase, epoch_loss))
                self.model.eval()
                acc_result = self.get_accuracy(inputs, labels)
                print(acc_result)
                self.model.train()

            # deep copy the model
            if phase == 'val' and (best_loss == None or epoch_loss < best_loss):
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), best_path)
                print(f'The best model has been saved to {best_path} ...')

            torch.save(model.state_dict(), current_path)
            print(f'Current mode has been saved to {current_path} ...')

            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val loss: {:4f}'.format(best_loss))

        # load best model weights
        model.load_state_dict(best_model_wts)
        self.model = model

# model = keyboard_detection_net()
# model.train()
