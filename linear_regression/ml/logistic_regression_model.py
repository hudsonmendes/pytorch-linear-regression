import torch


class LogisticRegressionModel(torch.nn.Module):

    def __init__(self, input_dim, output_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(
            in_features=input_dim, out_features=output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)
