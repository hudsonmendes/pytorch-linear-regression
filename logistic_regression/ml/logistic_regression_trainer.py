from pandas.io.formats.format import CategoricalFormatter
import torch
import numpy as np
import pandas as pd
from torch.optim import optimizer
from tqdm import trange

from .logistic_regression_model import LogisticRegressionModel

class LogisticRegressionTrainer:
    model: LogisticRegressionModel
    learning_rate: float

    def __init__(self, model: LogisticRegressionModel, learning_rate: float):
        self.model = model
        self.criterion = torch.nn.MSELoss()
        self.optimiser = torch.optim.SGD(self.model.parameters(), lr=learning_rate)

    def train(self, dataset: pd.DataFrame, epochs: int):
        
        for _ in trange(epochs, desc='training'):
            for _, (x, y) in dataset.iterrows():
                x = torch.autograd.Variable(torch.from_numpy(np.array([x], dtype=np.float32)))
                y = torch.autograd.Variable(torch.from_numpy(np.array([y], dtype=np.float32)))
                self.optimiser.zero_grad()
                y_hat = self.model(x)
                loss = self.criterion(y_hat, y)
                loss.backward()
                self.optimiser.step()