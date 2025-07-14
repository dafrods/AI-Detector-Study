
"""
A training framework for classification tasks.
"""

import os

from numpy import sum
from torch import sigmoid, device
from torch.nn import Module
from torch.nn.functional import softmax
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from PIL import Image

import optuna
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from typing import Optional, Literal

class ClassificationModelTrainer:

    def __init__(self,
                 model: Module,
                 training_set: Dataset,
                 validation_set: Dataset,
                 batch_size: int,
                 minimising_criterion: _Loss,
                 optimiser: Optimizer,
                 pin_memory=False,
                 device=device('cpu')) -> None:
        """
        Initialise a classification model training module.

        :param model: The model to train.
        :param training_set: The set of training data.
        :param validation_set: The set of validation data.
        :param batch_size: The batch size for training.
        :param minimising_criterion: The loss function.
        :param optimiser: The algorithm to perform minimisation task.
        """
        self._device = device
        self._model = model.to(self._device)

        self._train_loader =        DataLoader(dataset=training_set,   batch_size=batch_size, shuffle=True, pin_memory=pin_memory)
        self._validation_loader =   DataLoader(dataset=validation_set, batch_size=batch_size, shuffle=True, pin_memory=pin_memory)

        self._minimising_criterion = minimising_criterion
        self._optimiser = optimiser

        self.training_loss = []
        self.validation_acc = []

    def train_model(self, n_epochs) -> None:
        """
        Perform the model training.

        :param n_epochs: The number of training epochs to run.
        """

        # Training through the epochs.
        for epoch in range(n_epochs):
            loss_sublist = []
            # Training Process
            for x, y in self._train_loader:
                x, y = x.to(self._device), y.to(self._device)
                self._model.train()
                z = self._model(x)
                loss = self._minimising_criterion(z, y)
                loss_sublist.append(loss.data.item())
                loss.backward()
                self._optimiser.step()
                self._optimiser.zero_grad()

            self.training_loss.append(sum(loss_sublist))
            
            # Validation Process
            correct = 0
            n_test = 0
            for x_test, y_test in self._validation_loader:
                x_test, y_test = x_test.to(self._device), y_test.to(self._device)
                self._model.eval()
                z = softmax(self._model(x_test), dim=1)
                y_hat = z.data
                correct += (y_hat == y_test).sum().item()
                n_test += y_hat.shape[0]

            accuracy = correct / n_test
            self.validation_acc.append(accuracy)



class ImageDataset(Dataset):

    KINDS = ['Validation', 'Train', 'Test']

    def __init__(self, 
                 path:Optional[str] = None, 
                 limit:Optional[int] = None, 
                 kind : Literal['Validation', 'Train', 'Test'] = "Train", 
                 transform=None
        ):
        
        # Start Parameter Processing
        if path == None:
            self.dataset_root = 'Dataset'
        
        if kind in ImageDataset.KINDS:
            self.kind = kind
        else:
            raise ValueError(f"Value of Kind {kind} Kind must be one of the following: 'Validation', 'Train' (DEFAULT), 'Test' ")
        
        self.transform = transform
        # End Parameter Processing
        # print(os.getcwd())
        # print(Path(__file__).parent)
        # print(f"{self.dataset_root}/{self.kind}/Dataset.csv")

        self.cwd = Path(__file__).parent
        self.datapath = os.path.join(self.cwd, self.dataset_root, self.kind)

        path = os.path.join(self.cwd, self.dataset_root, f'{self.kind}.csv')

        if not os.path.exists(path):
            print("Creating CSV")
            self._create_csv(os.path.join(self.cwd, self.dataset_root, self.kind))

        # print(path)

        self.dataset = pd.read_csv(path)
        self.len = self.dataset.shape[0]
        # Produce a csv with n values


    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        path, classification = self.dataset.iloc[idx]
        img = Image.open(os.path.join(self.datapath,path))

        if self.transform:
            return self.transform(img), classification
        return img, classification
        

    def _create_csv(self, path):
        real = pd.DataFrame(os.listdir(os.path.join(path,"Real")))
        fake = pd.DataFrame(os.listdir(os.path.join(path,"Fake")))

        real = "Real/" + real
        fake = "Fake/" + fake

        real["Class"] = 0
        fake["Class"] = 1

        dataset = pd.concat([real, fake])

        dataset = dataset.rename(columns={0:"Path"})
        # dataset = dataset.reset_index(drop=True)

        dataset.to_csv(f"{self.cwd}/{self.dataset_root}/{self.kind}.csv", index=False, header=True, encoding="utf8")


def main():
    test = ImageDataset()
    print(len(test))
    plt.imshow(test[5][0])
    plt.show()

if __name__ == "__main__":
    main()