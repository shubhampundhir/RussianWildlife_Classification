import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.pool_large = nn.MaxPool2d(kernel_size=4, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)

        #Flattended size
        flattended_size = 128 * 14 * 14
        self.fc = nn.Linear(flattended_size, num_classes)

    def forward(self, x):
        # print("Input batch size:", x.size(0))
        x = self.conv1(x)
        # print("After conv1:", x.size())
        x = F.relu(x)
        x = self.pool_large(x)
        # print("After pool_large:", x.size())

        x = self.conv2(x)
        # print("After conv2:", x.size())
        x = F.relu(x)
        x = self.pool(x)
        # print("After pool:", x.size())

        x = self.conv3(x)
        # print("After conv3:", x.size())
        x = F.relu(x)
        x = self.pool(x)
        # print("After pool:", x.size())

        x = x.view(x.size(0), -1)
        # print("After flattening: ", x.size())
        x = self.fc(x)
        # print(x.size())
        return x
    
# Create an instance of the model CNN
model = Net()
# print(model)

