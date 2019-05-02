if __name__ == "__main__":
    print("Your are trying to run this .py directly, but you should not.")
    print("A wrapper API of this model (feeding data, etc.) is in the parent folder.")
    print("You should run a notebook in the parent folder as well.")
    exit(1)

import torch
import torch.nn as nn
import torch.optim as optim

from .model_wrapper import ModelWrapper
from .separate import white_key_height, white_key_width, black_key_width, black_key_height, img_height, img_width, white_key_width_bundle, black_key_width_bundle

white_fc_in = (white_key_width // 2 // 2) * (white_key_height // 2 // 2) * 32
black_fc_in = (black_key_width // 2 // 2) * (black_key_height // 2 // 2) * 32
w_bundle_fc_in = (white_key_width_bundle // 2 // 2) * (white_key_height // 2 // 2) * 32
b_bundle_fc_in = (black_key_width_bundle // 2 // 2) * (black_key_height // 2 // 2) * 32
all_fc_in = (img_width // 2 // 2 // 2 // 2) * (img_height // 2 // 2 // 2 // 2) * 32


class Flatten(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_2layer_model(linear_in):
    model_struct = torch.nn.Sequential(
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
        Flatten(),
        torch.nn.Linear(linear_in, 512),
        torch.nn.Linear(512, 1),
        torch.nn.Sigmoid()
    )
    return ModelWrapper(
        model_struct,
        torch.nn.BCELoss,
        torch.optim.Adam
    )

def get_3layer_model(linear_in):
    model_struct = torch.nn.Sequential(
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
        Flatten(),
        torch.nn.Linear(linear_in, 512),
        torch.nn.Linear(512, 1),
        torch.nn.Sigmoid()
    )
    return ModelWrapper(
        model_struct,
        torch.nn.BCELoss,
        torch.optim.Adam
    )

def get_full_model(linear_in):
    model_struct = torch.nn.Sequential(
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
        torch.nn.Linear(linear_in, 1024),
        torch.nn.Linear(1024, 88),
        torch.nn.Sigmoid()
    )
    return ModelWrapper(
        model_struct,
        torch.nn.BCELoss,
        torch.optim.Adam
    )

white_key_model = get_2layer_model(white_fc_in)
black_key_model = get_2layer_model(black_fc_in)
white_bundle_model = get_3layer_model(w_bundle_fc_in)
black_bundle_model = get_3layer_model(b_bundle_fc_in)
keyboard_model = get_full_model(all_fc_in)
