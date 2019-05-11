import torch
import torch.nn as nn
import torch.optim as optim

import time
import copy
import numpy as np
from torch.utils.tensorboard import SummaryWriter

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

    def evaluate(self, X, threshold=None):
        '''
        please use NCHW format
        '''
        if not isinstance(X, torch.Tensor):
            X = torch.Tensor(X)
            if torch.cuda.is_available():
                X = X.cuda()
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X)
            outputs = torch.squeeze(outputs)
            if threshold is not None:
                sm_mask = outputs < threshold
                lg_mask = outputs >= threshold
                outputs[sm_mask] = 0
                outputs[lg_mask] = 1
                return outputs.type(torch.ByteTensor)
            return outputs

    def get_accuracy_matrix(self, X, y, threshold=0.5):
        """
        Returns a 2*2 matrix of two values:
        """
        y_pred = self.evaluate(X, threshold=0.5).cpu()
        y = y.cpu().to(torch.uint8)
        acc = np.empty((2, 2), dtype=int)
        for i in (0, 1):
            for j in (0, 1):
                acc[i,j] = torch.sum((y_pred == i) & (y == j)).tolist()
        return acc

    @staticmethod
    def evaluate_accuracy_matrix(acc):
        try:
            precision = acc[1,1] / (acc[1,1] + acc[1,0])
        except ZeroDivisionError:
            precision = -1
        try:
            recall = acc[1,1] / (acc[1,1] + acc[0,1])
        except ZeroDivisionError:
            recall = -1
        return precision, recall

    def train(
            self,
            dataset,
            size='single',
            color='white',
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
        model = self.model
        criterion = self.loss_fn()
        optimizer = self.optim(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=decay_every, gamma=0.05)

        writer = SummaryWriter(f'{size}_{color}_{time.ctime().replace(" ", "_").replace(":", "_")}')
        since = time.time()

        best_model_wts = copy.deepcopy(model.state_dict())
        best_loss = None

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
                accuracy_matrix = np.zeros((2, 2), dtype=int)
                max_num_for_this_epoch = max_num if phase == 'train' else -1
                dbatch = dataset.data_batch(type=phase, size=size, color=color, 
                                            batch_size=batch_size,
                                            need_velocity=False,
                                            NCHW=True,
                                            max_num=max_num_for_this_epoch)
                # Iterate over data.
                for inputs, labels in dbatch:
                    inputs = torch.Tensor(inputs)
                    labels = torch.Tensor(labels)
                    if torch.cuda.is_available():
                        inputs = inputs.cuda()
                        labels = labels.cuda()

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    outputs = model(inputs)
                    outputs = torch.squeeze(outputs)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    # statistics
                    running_loss += loss.item() * batch_size
                    accuracy_matrix += self.get_accuracy_matrix(inputs, labels)
                    if phase == 'train':
                        model.train()  # Set model to training mode
                    else:
                        model.eval()  # Set model to evaluate mode

                    # free unoccupied memory
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                epoch_loss = running_loss / dbatch.max_num
                print('{} Loss: {:.4f}'.format(
                    phase, epoch_loss))

                precision, recall = self.evaluate_accuracy_matrix(accuracy_matrix)
                print('Precision: %.2f' % precision)
                print('Recall   : %.2f' % recall)

                writer.add_scalar(f'{phase}_loss', epoch_loss, epoch)
                writer.add_scalar(f'{phase}_accuracy/precision', precision)
                writer.add_scalar(f'{phase}_accuracy/recall', recall)

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
