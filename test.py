import os

# Set the cache path for torchvision
os.environ['TORCH_HOME'] = r'models/pretrained/'


from PIL import Image
from matplotlib import pyplot as plt

from models import *
from torchvision import transforms

from torch import nn, tensor
#  instanciate models to a list then make it predict

# t = transforms.Resize((224,224))

t = transforms.Compose(
    [transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
)

im_path = r"Dataset\Test\Fake\fake_0.jpg"

def create_models():
    models = [
        VGG19,
        DenseNet121,
        ResNet50,
        GoogLeNet,
        EfficientNetB0
    ]
    return models

def test_models(models, img):
    outputs = []
    for model in models:
        model.eval()
        z = model(img)
        outputs.append(z)
    return outputs

def load_image(path):
    img = Image.open(path)
    return img

def show_image(img):
    plt.imshow(img)
    plt.axis('off')
    plt.show()

def main():
    img = load_image(im_path)
    # print(img.size)
    # show_image(img)
    # img_ = t(img)
    # print(img_.size)
    # show_image(img_)

    # model = VGG19(nn.Sequential(nn.Flatten(),nn.Linear(49,1), nn.Sigmoid()),
    #               pretrained=True)

    # model = VGG19()
    # model = ResNet50()
    # model = GoogLeNet()
    model = EfficientNetB0()
    # model = DenseNet121()
    model.eval()

    img_ = t(img)
    img_ = img_.unsqueeze(0)
    # print(img_)
    z = model(img_)

    print(z.shape)
    print(z)

    # models = create_models()
    # outputs = test_models(models, img_)

    # print(outputs)

if __name__ == "__main__":
    main()