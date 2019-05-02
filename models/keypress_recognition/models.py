if __name__ == "__main__":
    print("Your are trying to run this .py directly, but you should not.")
    print("A wrapper API of this model (feeding data, etc.) is in the parent folder.")
    print("You should run a notebook in the parent folder as well.")
    exit(1)

import torch
import torch.nn as nn
import torch.optim as optim

from ..model_wrapper import ModelWrapper
from seperate import white_key_height, white_key_width, black_key_width, black_key_height, img_height, img_width

white_fc_in = (white_key_width //2 //2) * (white_key_height //2 //2) * 32
black_fc_in = (black_key_width //2 //2) * (black_key_height //2 //2) * 32
all_fc_in = (img_width//2//2//2//2) * (img_height//2//2//2//2)*32

class Flatten(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


white_key_struct = torch.nn.Sequential(
    torch.nn.Conv2d(1, 16, 3, padding=1),
    torch.nn.ReLU(),
    torch.nn.BatchNorm2d(16),
    torch.nn.Conv2d(16, 16, 3, padding=1),
    torch.nn.ReLU(),
    torch.nn.BatchNorm2d(16),
    torch.nn.MaxPool2d(2),
    torch.nn.Conv2d(16, 32, 3, padding=1),
    torch.nn.ReLU(),
    torch.nn.BatchNorm2d(16),
    torch.nn.Conv2d(32, 32, 3, padding=1),
    torch.nn.ReLU(),
    torch.nn.BatchNorm2d(32),
    torch.nn.MaxPool2d(2),
    Flatten(),
    torch.nn.Linear(white_fc_in, 512),
    torch.nn.Linear(512, 1),
    torch.nn.Sigmoid()
)


white_key_model = ModelWrapper(
    white_key_struct,
    torch.nn.BCELoss,
    torch.optim.Adam
)

black_key_struct = torch.nn.Sequential(
    torch.nn.Conv2d(1, 16, 3, padding=1),
    torch.nn.ReLU(),
    torch.nn.BatchNorm2d(16),
    torch.nn.Conv2d(16, 16, 3, padding=1),
    torch.nn.ReLU(),
    torch.nn.BatchNorm2d(16),
    torch.nn.MaxPool2d(2),
    torch.nn.Conv2d(16, 32, 3, padding=1),
    torch.nn.ReLU(),
    torch.nn.BatchNorm2d(16),
    torch.nn.Conv2d(32, 32, 3, padding=1),
    torch.nn.ReLU(),
    torch.nn.BatchNorm2d(32),
    torch.nn.MaxPool2d(2),
    Flatten(),
    torch.nn.Linear(black_fc_in, 512),
    torch.nn.Linear(512, 1),
    torch.nn.Sigmoid()
)


black_key_model = ModelWrapper(
    black_key_struct,
    torch.nn.BCELoss,
    torch.optim.Adam
)


keyboard_model = torch.nn.Sequential(
    torch.nn.Conv2d(1, 16, 3, padding=1),
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
    torch.nn.Conv2d(32, 64, 3, padding=1),
    torch.nn.ReLU(),
    torch.nn.BatchNorm2d(64),
    torch.nn.Conv2d(64, 64, 3, padding=1),
    torch.nn.ReLU(),
    torch.nn.BatchNorm2d(64),
    torch.nn.MaxPool2d(2),
    torch.nn.Conv2d(64, 128, 3, padding=1),
    torch.nn.ReLU(),
    torch.nn.BatchNorm2d(128),
    torch.nn.Conv2d(128, 128, 3, padding=1),
    torch.nn.ReLU(),
    torch.nn.BatchNorm2d(128),
    torch.nn.MaxPool2d(2),
    Flatten(),
    torch.nn.Linear(all_fc_in, 1024),
    torch.nn.Linear(1024, 88),
    torch.nn.Sigmoid()
)