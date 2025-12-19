import torch.nn as nn
import torch.nn.functional as F

class CNN_CIFAR(nn.Module):
    def __init__(self):
        super(CNN_CIFAR, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=2),
            nn.GroupNorm(num_groups=2, num_channels=32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.1),

            # Block 2
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=2),
            nn.GroupNorm(num_groups=2, num_channels=32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.1),

            # Block 3
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=2),
            nn.GroupNorm(num_groups=2, num_channels=32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.1),

            # Block 4
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=2),
            nn.GroupNorm(num_groups=2, num_channels=32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.1),
        )

        self.fc = nn.Linear(32 * 3 * 3, 10)

    def forward(self, inputs):
        # CIFAR-10 images are 32x32 with 3 channels
        x = inputs.view(-1, 3, 32, 32)
        
        x = self.features(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Output layer
        x = self.fc(x)
        return x
def cnn_cifar10(args):
    return CNN_CIFAR(class_num=args.class_num)

def cnn_cifar100(args):
    return CNN_CIFAR(class_num=args.class_num)
