import torch
import torch.nn as nn
import loader
import time

start_time = time.perf_counter()

cfg_vgg16 = [64, 64, 'M', 
            128, 128, 'M', 
            256, 256, 256, 'M',
            512, 512, 512, 'M', 
            512, 512, 512, 'M']

def make_layers(cfg):
    layers = []
    in_channels = 3

    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            layers += [
                nn.Conv2d(in_channels, v, kernel_size=3, padding=1),
                nn.BatchNorm2d(v),
                nn.ReLU(inplace=True)
            ]
            in_channels = v

    return nn.Sequential(*layers)


class VGG(nn.Module):
    def __init__(self, features, num_classes=10):

        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512 * 2 * 2, 1024), 
            nn.ReLU(True),

            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    

def vgg16(num_classes=10):
    return VGG(make_layers(cfg_vgg16), num_classes=num_classes)

model = vgg16(num_classes=10)

train_loader, test_loader = loader.get_dataset()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)


for epoch in range(30):
    for imgs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
    end_time = time.perf_counter()
    
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.6f} seconds")
    
