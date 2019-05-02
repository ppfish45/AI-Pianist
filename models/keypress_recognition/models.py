import torch
from ..model_wrapper import ModelWrapper


class Flatten(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)


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
    torch.nn.Linear(3328, 512),
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
    torch.nn.Linear(1088, 512),
    torch.nn.Linear(512, 1),
    torch.nn.Sigmoid()
)


black_key_model = ModelWrapper(
    black_key_struct,
    torch.nn.BCELoss,
    torch.optim.Adam
)
