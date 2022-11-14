import torch
import torch.nn as nn


class LearnFocal(nn.Module):
    def __init__(self, H, W):
        super(LearnFocal, self).__init__()
        # self.H = torch.tensor(
        #     H, dtype=torch.float32).cuda()
        # self.W = torch.tensor(
        #     W, dtype=torch.float32).cuda()
        self.H = torch.tensor(
            420.0, dtype=torch.float32).cuda()
        self.W = torch.tensor(
            420.0, dtype=torch.float32).cuda()
        self.H_temp = torch.tensor(
            300.0, dtype=torch.float32).cuda()
        self.W_temp = torch.tensor(
            105.0, dtype=torch.float32).cuda()
        self.fx = nn.Parameter(torch.tensor(
            1.0, dtype=torch.float32), requires_grad=True)  # (1, )
        self.fy = nn.Parameter(torch.tensor(
            1.0, dtype=torch.float32), requires_grad=True)  # (1, )
        #self.layer = nn.Linear(4, 4, bias=False)

    def forward(self):
        # order = 2, check our supplementary.
        fxfy = torch.stack([self.fx**2 * self.W, self.fy **
                           2 * self.H, self.H_temp, self.W_temp])
        # x = torch.stack([self.fx**2*self.H, self.fy**2*self.H, self.W, self.H])
        # fxfy = self.layer(x)
        return fxfy
