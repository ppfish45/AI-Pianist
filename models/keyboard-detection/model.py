import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

import time
import copy
import tqdm

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

    def load_best_model(self, path='best_model.tar'):
        print(f'Trying to load model from {path} ...')
        self.model.load_state_dict(torch.load(path))

    def train(self, batch_size=64, learning_rate=1e-3, num_epochs=5, max_num=-1, path='best_model.tar'):

        model = self.model    
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.05)

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

                for inputs, labels in dataset.data_batch(type=phase, batch_size=batch_size, max_num=max_num_for_this_epoch):

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
                    torch.cuda.empty_cache()
                        
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
                torch.save(model.state_dict(), path)
                print(f'The best model has been saved to {path} ...')

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
