import torch
import torch.nn.functional as F

# 全连接层
class FCModel(torch.nn.Module):
    def __init__(self,in_features,block_size):
        super(FCModel, self).__init__()
        self.conv1 = torch.nn.Conv1d(in_channels=in_features, out_channels=32, kernel_size=3, padding=1)
        self.fc1 = torch.nn.Linear(32 * block_size, 1) 

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = F.relu(self.conv1(x))
        x = x.view(x.size(0), -1)
        x = torch.sigmoid(self.fc1(x))
        return x