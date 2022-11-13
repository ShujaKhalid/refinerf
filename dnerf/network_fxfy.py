import torch
import torch.nn as nn


class LearnFocal(nn.Module):
    def __init__(self, H, W):
        super(LearnFocal, self).__init__()
        self.H = nn.Parameter(torch.tensor(
            H, dtype=torch.float32), requires_grad=True)
        self.W = nn.Parameter(torch.tensor(
            W, dtype=torch.float32), requires_grad=True)
        self.fx = nn.Parameter(torch.tensor(
            1.0, dtype=torch.float32), requires_grad=True)  # (1, )
        self.fy = nn.Parameter(torch.tensor(
            1.0, dtype=torch.float32), requires_grad=True)  # (1, )
        self.layer = nn.Linear(4, 4, bias=False)

    def forward(self):
        # order = 2, check our supplementary.
        #fxfy = torch.stack([self.fx**2 * self.W, self.fy**2 * self.H])
        x = torch.stack([self.fx**2*self.W, self.fy**2*self.H, self.W, self.H])
        fxfy = self.layer(x)
        return fxfy
