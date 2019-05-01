import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

import time
import copy
import tqdm
import cv2
import numpy as np

from ipywidgets import IntProgress
from IPython.display import display

import dataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class keyboard_detection_net():
    def __init__(self):
        self.model = models.resnet18(pretrained=True)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features=in_features, out_features=8, bias=True)
        if torch.cuda.is_available():
            self.model.to(device)

    def load_model(self, path):
        print(f'Trying to load model from {path} ...')
        self.model.load_state_dict(torch.load(path))
        if torch.cuda.is_available():
            self.model.to(device)
        print('done')

    def evaluate(self, X):
        '''
        please use NCHW format
        '''
        with torch.no_grad():
            self.model.eval()
            inputs = torch.Tensor(X)
            inputs = inputs.to(device)
            outputs = torch.squeeze(self.model(inputs))
            outputs = torch.reshape(outputs, [-1, 4, 2])
            out = outputs.cpu().numpy()
            transformed_image = []
            for i in range(X.shape[0]):
                width = 884
                height = 106
                img = np.transpose(X[i], [1, 2, 0])
                dst = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
                M = cv2.getPerspectiveTransform(out[i], dst)
                result = cv2.warpPerspective(img, M, (width, height))
                transformed_image.append(result)
            return (out, transformed_image)

    def train(self, batch_size=64, learning_rate=1e-3, num_epochs=5, max_num=-1,
              best_path='keyboard_model_best.tar', 
              current_path='keyboard_model_latest.tar',
              decay_every=10,
              save_model=True,
              dirs=[0]):

        model = self.model    
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=decay_every, gamma=0.05)

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
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                # Iterate over data.

                max_num_for_this_epoch = max_num if phase == 'train' else -1

                total = dataset.get_num_of_data(phase) if max_num == -1 else max_num

                bar = IntProgress(max=total)

                display(bar)

                for inputs, labels in dataset.data_batch(type=phase, batch_size=batch_size,
                                                         max_num=max_num_for_this_epoch,
                                                         dirs=dirs):

                    inputs = torch.Tensor(inputs)
                    labels = torch.Tensor(labels)

                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    outputs = model(inputs)
                    labels = torch.reshape(labels, [-1, 8])
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
                    else:
                        torch.cpu.empty_cache()
                        
                    bar.value += batch_size
                    bar.description = f'{bar.value} / {total}'

                bar.close()

                epoch_loss = running_loss / dataset.get_num_of_data(phase)

                print('{} Loss: {:.4f}'.format(
                    phase, epoch_loss))

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
