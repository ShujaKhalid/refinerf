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
        # self.layer = nn.Linear(2, 2, bias=False)

    def forward(self):
        # order = 2, check our supplementary.

        PREDICT_FXFY = True

        # predict fxfy
        fxfy = torch.stack([self.fx**2 * self.W if PREDICT_FXFY else self.H,
                            self.fy**2 * self.W if PREDICT_FXFY else self.W,
                            self.H_temp if PREDICT_FXFY else self.fx ** 2 * self.W,
                            self.W_temp if PREDICT_FXFY else self.fy ** 2 * self.H
                            ])

        # Model test
        # x = torch.stack([self.H, self.W])
        # fxfy = self.layer(x)
        # fxfy = torch.stack([fxfy[0], fxfy[1], self.H_temp, self.W_temp])

        return fxfy
