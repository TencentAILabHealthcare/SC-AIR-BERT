import torch

class FCModel(torch.nn.Module):
    def __init__(self,in_features):
        super(FCModel, self).__init__()

        self.fc1 = torch.nn.Linear(in_features=in_features, out_features=1)

    def forward(self, input):
        score = self.fc1(input)

        result = torch.sigmoid(score)

        return result
