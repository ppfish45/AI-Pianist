import torch
from ..model_factory import ModelWrapper

class Flatten(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)

loss_fn = torch.nn.BCELoss()

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


white_key_optim = torch.optim.Adam(white_key_struct.parameters(), lr=1e-4)
white_key_scheduler = torch.optim.lr_scheduler.StepLR(white_key_optim, step_size=10, gamma=0.05)

white_key_model = ModelWrapper(
    white_key_struct,
    loss_fn,
    white_key_optim,
    white_key_scheduler
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

black_key_optim = torch.optim.Adam(black_key_struct.parameters(), lr=1e-4)
black_key_scheduler = torch.optim.lr_scheduler.StepLR(black_key_optim, step_size=10, gamma=0.05)

black_key_model = ModelWrapper(
    black_key_struct,
    loss_fn,
    black_key_optim,
    black_key_scheduler
)
