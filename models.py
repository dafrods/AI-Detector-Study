import torch
import torchvision.models as models

from typing import Optional

MODELS = [
    "CustomCNN",
    "VGG19",
    "DenseNet121",
    "ResNet50",
    "GoogLeNet",
    "EfficientNet-B0",
]

BINARY_CLASSIFICATION = 1

DENSENET121_FEATURE_SIZE = 512*4
VGG19_FEATURE_SIZE = 512*7*7
RESNET50_FEATURE_SIZE = 512 * 4
GOOGLENET_FEATURE_SIZE = 1024
EFFICIENTNETB0_CONVOUT_SIZE = 320 * 4

class CustomCNN(torch.nn.Module):
    def __init__(self, in_features=256, out_features=1):
        super(CustomCNN,self).__init__()
        self.model = torch.nn.Linear(in_features=in_features,out_features=out_features)

    def forward(self,x):
        pass

class VGG19(torch.nn.Module):
    def __init__(self,
                 classifier : Optional[torch.nn.Sequential] = None,
                 dropout : float = .5,
                 freeze=True ,
                 pretrained=False
        ):
        super(VGG19,self).__init__()

        if classifier == None:
            classifier = torch.nn.Sequential(
                    torch.nn.Linear(VGG19_FEATURE_SIZE, 4096),
                    torch.nn.ReLU(True),
                    torch.nn.Dropout(p=dropout),
                    torch.nn.Linear(4096, 4096),
                    torch.nn.ReLU(True),
                    torch.nn.Dropout(p=dropout),
                    torch.nn.Linear(4096, BINARY_CLASSIFICATION),
            )

        weights = None

        if pretrained:
            weights = models.VGG19_Weights.DEFAULT

        self.model = models.vgg19(weights=weights)

        # Freeze Layers
        if not freeze:
            for param in self.model.parameters():
                param.requires_grad = False

        self.model.classifier = classifier


class DenseNet121(torch.nn.Module):
    def __init__(self,
                 classifier: Optional[torch.nn.Linear] = None,
                 freeze = True,
                 pretrained = False
        ):

        super(DenseNet121,self).__init__()

        if classifier == None:
            classifier = torch.nn.Linear(DENSENET121_FEATURE_SIZE, BINARY_CLASSIFICATION)

        weights = None

        if pretrained:
            weights = models.VGG16_Weights.DEFAULT

        self.model = models.densenet121(weights=weights)

        # Freeze Layers
        if not freeze:
            for param in self.model.parameters():
                param.requires_grad = False

        self.model.classifier = classifier
    

class ResNet50(torch.nn.Module):
    def __init__(self,
                 classifier: Optional[torch.nn.Linear] = None,
                 freeze = True,
                 pretrained = False
        ):
        
        super(ResNet50,self).__init__()

        if classifier == None:
            classifier = torch.nn.Linear(RESNET50_FEATURE_SIZE, BINARY_CLASSIFICATION)

        weights = None

        if pretrained:
            weights = models.ResNet50_Weights.DEFAULT
        
        self.model = models.resnet50(weights=weights)

        # Freeze Layers
        if not freeze:
            for param in self.model.parameters():
                param.requires_grad = False
        
        self.model.fc = classifier

    
class GoogLeNet(torch.nn.Module):
    def __init__(self, 
                 classifier: Optional[torch.nn.Linear] = None,
                 freeze = True,
                 pretrained = False
        ):
        super(GoogLeNet,self).__init__()

        if classifier == None:
            classifier = torch.nn.Linear(GOOGLENET_FEATURE_SIZE, BINARY_CLASSIFICATION)

        weights = None

        if pretrained:
            weights = models.GoogLeNet_Weights.DEFAULT
        
        self.model = models.googlenet(weights=weights)

        # Freeze Layers
        if not freeze:
            for param in self.model.parameters():
                param.requires_grad = False
        
        self.model.fc = classifier


class EfficientNetB0(torch.nn.Module):
    def __init__(self, 
                 classifier: torch.nn.Linear,
                 dropout = 0.5,
                 freeze = True, 
                 pretrained = False
        ):
        super(EfficientNetB0,self).__init__()

        if classifier == None:
            classifier = torch.nn.Sequential(
                         torch.nn.Dropout(p=dropout, inplace=True),
                         torch.nn.Linear(EFFICIENTNETB0_CONVOUT_SIZE, BINARY_CLASSIFICATION),
            )

        weights = None

        if pretrained:
            weights = models.EfficientNet_B0_Weights.DEFAULT
        
        self.model = models.efficientnet_b0(weights=weights)

        # Freeze Layers
        if not freeze:
            for param in self.model.parameters():
                param.requires_grad = False
        
        self.model.fc = classifier
