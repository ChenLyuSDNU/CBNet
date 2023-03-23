import torch
import torch.nn as nn
import torchsummary


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.pool = nn.AdaptiveAvgPool2d(7)
        self.classifier = nn.Sequential(
            nn.Linear(6272, 2048, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(2048, 1000, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(1000, 3, bias=True)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        out = self.classifier(x)
        return out


def mymodel():
    model = MyModel()
    torch.load('./checkpoints/mymodel/model_best.pth_leaf_94.52.tar')
    return model

# model = MyModel()
# # print(model)
# torchsummary.summary(model, input_size=[(3, 650, 650)], device="cpu")