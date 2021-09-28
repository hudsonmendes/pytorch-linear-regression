import torch

#TODO: implement logistic regresion
class LogisticRegression(torch.nn.Module):

    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(in_features=input_dim, out_features=output_dim)

    def forward(self, x):
        return self.linear(x)