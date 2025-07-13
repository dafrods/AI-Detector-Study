import torch
import torchvision.models as models

from typing import Optional

__all__ = [
    "CustomCNN",
    "VGG19",
    "DenseNet121",
    "ResNet50",
    "GoogLeNet",
    "EfficientNetB0",
]

BINARY_CLASSIFICATION = 1

DENSENET121_FEATURE_SIZE = 512*2
VGG19_FEATURE_SIZE = 512*7*7
RESNET50_FEATURE_SIZE = 512 * 4
GOOGLENET_FEATURE_SIZE = 1024
EFFICIENTNETB0_CONVOUT_SIZE = 320 * 4

class CustomCNN(torch.nn.Module):
    def __init__(self, in_features=256, out_features=1):
        super().__init__()
        self.model = torch.nn.Linear(in_features=in_features,out_features=out_features)

    def forward(self,x):
        pass

class VGG19(torch.nn.Module):
    def __init__(self,
                 classifier : Optional[torch.nn.Linear] = None,
                 dropout : float = .5, # No longer in use
                 freeze=True ,
                 pretrained=False
        ):
        super().__init__()

        # Change the last layer instead to preserve the other parameters
        # if classifier == None:
        #     classifier = torch.nn.Sequential(
        #             torch.nn.Linear(VGG19_FEATURE_SIZE, 4096),
        #             torch.nn.ReLU(True),
        #             torch.nn.Dropout(p=dropout),
        #             torch.nn.Linear(4096, 4096),
        #             torch.nn.ReLU(True),
        #             torch.nn.Dropout(p=dropout),
        #             torch.nn.Linear(4096, BINARY_CLASSIFICATION),
        #     )

        if classifier == None:
            classifier = torch.nn.Linear(4096, BINARY_CLASSIFICATION)

        weights = models.VGG19_Weights.DEFAULT if pretrained else None

        self.model = models.vgg19(weights=weights)

        # Freeze Layers
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

        self.model.classifier[-1] = classifier # Change last layer instead

    def forward(self, x):
        x = self.model(x)
        return x



class DenseNet121(torch.nn.Module):
    def __init__(self,
                 classifier: Optional[torch.nn.Linear] = None,
                 freeze = True,
                 pretrained = False
        ):

        super().__init__()

        if classifier == None:
            classifier = torch.nn.Linear(DENSENET121_FEATURE_SIZE, BINARY_CLASSIFICATION)

        weights = models.VGG16_Weights.DEFAULT if pretrained else None

        self.model = models.densenet121(weights=weights)

        # Freeze Layers
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

        self.model.classifier = classifier

    def forward(self, x):
        x = self.model(x)
        return x

class ResNet50(torch.nn.Module):
    def __init__(self,
                 classifier: Optional[torch.nn.Linear] = None,
                 freeze = True,
                 pretrained = False
        ):
        
        super().__init__()

        if classifier == None:
            classifier = torch.nn.Linear(RESNET50_FEATURE_SIZE, BINARY_CLASSIFICATION)

        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        
        self.model = models.resnet50(weights=weights)

        # Freeze Layers
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
        
        self.model.fc = classifier
    
    def forward(self, x):
        x = self.model(x)
        # x = torch.nn.Flatten(x,1) # nn.Module calls the original models forward(x) and this is detrimental
        # x = self.model.fc(x)
        return x

    
class GoogLeNet(torch.nn.Module):
    def __init__(self, 
                 classifier: Optional[torch.nn.Linear] = None,
                 freeze = True,
                 pretrained = False
        ):
        super().__init__()

        if classifier == None:
            classifier = torch.nn.Linear(GOOGLENET_FEATURE_SIZE, BINARY_CLASSIFICATION)

        weights = models.GoogLeNet_Weights.DEFAULT if pretrained else None
        
        self.model = models.googlenet(weights=weights)

        # Freeze Layers
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
        
        self.model.fc = classifier

    def forward(self, x):
        x = self.model(x)
        return x

class EfficientNetB0(torch.nn.Module):
    def __init__(self, 
                 classifier: Optional[torch.nn.Linear] = None,
                 dropout = 0.5,
                 freeze = True, 
                 pretrained = False
        ):
        super().__init__()

        # Change the last layer instead to preserve the other parameters
        # if classifier == None:
        #     classifier = torch.nn.Sequential(
        #                  torch.nn.Dropout(p=dropout, inplace=True),
        #                  torch.nn.Linear(EFFICIENTNETB0_CONVOUT_SIZE, BINARY_CLASSIFICATION),
        #     )

        if classifier == None:
                classifier = torch.nn.Linear(EFFICIENTNETB0_CONVOUT_SIZE, BINARY_CLASSIFICATION)

        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        
        self.model = models.efficientnet_b0(weights=weights)

        # Freeze Layers
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
        
        self.model.classifier[-1] = classifier

    def forward(self, x):
        x = self.model(x)
        return x