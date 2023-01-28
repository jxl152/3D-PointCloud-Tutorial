import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import pretty_errors


class PointNet(nn.Module):
  def __init__(self):
    super(PointNet, self).__init__()
    self.conv1 = nn.Conv1d(3, 64, 1)
    self.conv2 = nn.Conv1d(64, 128, 1)
    self.conv3 = nn.Conv1d(128, 1024, 1)
    self.fc1 = nn.Linear(1024, 512)
    self.fc2 = nn.Linear(512, 256)
    self.fc3 = nn.Linear(256, 40)

    self.bn1 = nn.BatchNorm1d(64)
    self.bn2 = nn.BatchNorm1d(128)
    self.bn3 = nn.BatchNorm1d(1024)
    self.bn4 = nn.BatchNorm1d(512)
    self.bn5 = nn.BatchNorm1d(256)

    self.relu = nn.ReLU(inplace=True)
    self.dropout = nn.Dropout(p=0.3)

  def forward(self, x):
    # TODO: use functions in __init__ to build network
    
    return x


if __name__ == "__main__":
  net = PointNet()
  sim_data = Variable(torch.rand(3, 3, 10000))
  out = net(sim_data)
  print('gfn', out.size())