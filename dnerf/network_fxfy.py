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
            H, dtype=torch.float32).cuda()
        self.W = torch.tensor(
            W, dtype=torch.float32).cuda()
        self.H_temp = torch.tensor(
            H/4, dtype=torch.float32).cuda()
        self.W_temp = torch.tensor(
            W/4, dtype=torch.float32).cuda()
        self.fx = nn.Parameter(torch.tensor(
            1.0, dtype=torch.float32), requires_grad=True)  # (1, )
        self.fy = nn.Parameter(torch.tensor(
            1.0, dtype=torch.float32), requires_grad=True)  # (1, )
        self.cx = nn.Parameter(torch.tensor(
            1.0, dtype=torch.float32), requires_grad=True)  # (1, )
        self.cy = nn.Parameter(torch.tensor(
            1.0, dtype=torch.float32), requires_grad=True)  # (1, )
        # self.layer = nn.Linear(4, 4, bias=False)

    def forward(self):
        # order = 2, check our supplementary.

        PREDICT_FXFY = True

        # predict fxfy
        fxfy = torch.stack([self.fx**2 * self.H if PREDICT_FXFY else self.H,
                            self.fy**2 * self.W if PREDICT_FXFY else self.W,
                            # self.H_temp if PREDICT_FXFY else self.fx ** 2 * self.W,
                            # self.W_temp if PREDICT_FXFY else self.fy ** 2 * self.H,
                            self.cx**2 * self.H_temp,
                            self.cy**2 * self.W_temp
                            ])

        # Model test
        # x = torch.stack([self.H, self.W, self.H_temp, self.W_temp])
        # fxfy = self.layer(x)
        # fxfy = torch.stack([fxfy[0], fxfy[1], fxfy[2], fxfy[3]])

        return fxfy
