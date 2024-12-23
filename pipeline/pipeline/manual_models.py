import torch
import torch.nn as nn
import torchvision.models as models

class ResNetWithEmbeddings(nn.Module):
    def __init__(self, num_classes=10, version='resnet18'):
        super(ResNetWithEmbeddings, self).__init__()
        self.resnet = getattr(models, version)(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.resnet.fc(x)
        return x

    def get_embeddings(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        embeddings = self.resnet.avgpool(x)
        embeddings = torch.flatten(embeddings, 1)
        return embeddings

class EfficientNetWithEmbeddings(nn.Module):
    def __init__(self, num_classes=10, version='b0'):
        super(EfficientNetWithEmbeddings, self).__init__()
        self.efficientnet = getattr(models, f'efficientnet_{version}')(pretrained=True)
        in_features = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.efficientnet.features(x)
        x = self.efficientnet.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.efficientnet.classifier(x)
        return x

    def get_embeddings(self, x):
        x = self.efficientnet.features(x)
        x = self.efficientnet.avgpool(x)
        embeddings = torch.flatten(x, 1)
        return embeddings