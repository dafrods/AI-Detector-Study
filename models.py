import torch
import torchvision.models as models

MODELS = [
    "CustomCNN",
    "VGG19",
    "DenseNet121",
    "ResNet50",
    "GoogLeNet",
    "EfficientNet-B4",
]

class CustomCNN(torch.nn.Module):
    def __init__(self, in_features=256, out_features=1):
        super(CustomCNN,self).__init__()
        self.model = torch.nn.Linear(in_features=in_features,out_features=out_features)

    def forward(self,x):
        pass

class VGG19(torch.nn.Module):
    def __init__(self,
                 classifier : torch.nn.Sequential,
                 freeze=True ,
                 pretrained=False
        ):
        super(VGG19,self).__init__()

        weights = None

        if pretrained:
            weights = models.VGG19_Weights.DEFAULT

        self.model = models.vgg19(weights=weights)

        # Freeze Layers
        if not freeze:
            for param in self.model.parameters():
                param.requires_grad = False

        self.model.classifier = classifier


    def forward(self,x):
        pass

class DenseNet121(torch.nn.Module):
    def __init__(self,
                 classifier: torch.nn.Linear = torch.nn.Linear(512*4, 1), # 
                 freeze = True,
                 pretrained = False
        ):

        super(DenseNet121,self).__init__()

        weights = None

        if pretrained:
            weights = models.VGG16_Weights.DEFAULT

        self.model = models.densenet121(weights=weights)

        # Freeze Layers
        if not freeze:
            for param in self.model.parameters():
                param.requires_grad = False

        self.model.classifier = classifier
    
    def forward(self,x):
        pass

class ResNet50(torch.nn.Module):
    def __init__(self,
                 classifier: torch.nn.Linear,
                 freeze = True,
                 pretrained = False
        ):
        
        super(ResNet50,self).__init__()

        weights = None

        if pretrained:
            weights = models.ResNet50_Weights.DEFAULT
        
        self.model = models.resnet50(weights=weights)

        # Freeze Layers
        if not freeze:
            for param in self.model.parameters():
                param.requires_grad = False
        
        self.model.fc = classifier

    
    def forward(self,x):
        pass
    
class GoogLeNet(torch.nn.Module):
    def __init__(self, 
                 classifier: torch.nn.Linear,
                 freeze = True,
                 pretrained = False
        ):
        super(GoogLeNet,self).__init__()

        weights = None

        if pretrained:
            weights = models.GoogLeNet_Weights.DEFAULT
        
        self.model = models.googlenet(weights=weights)

        # Freeze Layers
        if not freeze:
            for param in self.model.parameters():
                param.requires_grad = False
        
        self.model.fc = classifier

    
    def forward(self,x):
        pass

class EfficientNetB0(torch.nn.Module):
    def __init__(self, 
                 classifier: torch.nn.Linear,
                 freeze = True, 
                 pretrained = False
        ):
        super(EfficientNetB0,self).__init__()

        weights = None

        if pretrained:
            weights = models.EfficientNet_B0_Weights.DEFAULT
        
        self.model = models.efficientnet_b0(weights=weights)

        # Freeze Layers
        if not freeze:
            for param in self.model.parameters():
                param.requires_grad = False
        
        self.model.fc = classifier

    
    def forward(self,x):
        pass