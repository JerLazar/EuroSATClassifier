import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                            (0.229, 0.224, 0.225))
])

def get_dataset():
    train_dataset = datasets.EuroSAT(
        root="data",
        download=True,
        transform=transforms
    )

    train_size = int(0.8 * len(train_dataset))
    test_size = len(train_dataset) - train_size
    train_data, test_data = torch.utils.data.random_split(train_dataset, [train_size, test_size])

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
    return train_loader, test_loader