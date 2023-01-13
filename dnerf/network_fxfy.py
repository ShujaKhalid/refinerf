import torch
import torch.nn as nn


class LearnFocal(nn.Module):
    def __init__(self, H, W, noise_pct):
        super(LearnFocal, self).__init__()
        # self.H = torch.tensor(
        #     H, dtype=torch.float32).cuda()
        # self.W = torch.tensor(
        #     W, dtype=torch.float32).cuda()
        self.H = torch.tensor(
            H, dtype=torch.float32).cuda()
        self.W = torch.tensor(
            W, dtype=torch.float32).cuda()
        # self.fx = torch.tensor(
        #     416.0, dtype=torch.float32).cuda()
        # self.fy = torch.tensor(
        #     429.0, dtype=torch.float32).cuda()
        # self.H_temp = torch.tensor(
        #     H/2, dtype=torch.float32).cuda()
        # self.W_temp = torch.tensor(
        #     W/2, dtype=torch.float32).cuda()
        self.fact = torch.round(
            self.W/self.H) if self.W >= self.H else torch.round(self.H/self.W)
        if self.W == self.H:
            self.fact *= 2
        self.W_temp = torch.tensor(
            self.W/self.fact, dtype=torch.float32).cuda()
        self.H_temp = torch.tensor(
            self.H/self.fact, dtype=torch.float32).cuda()
        self.fx = nn.Parameter(torch.tensor(
            1.0, dtype=torch.float32), requires_grad=True)  # (1, )
        self.fy = nn.Parameter(torch.tensor(
            1.0, dtype=torch.float32), requires_grad=True)  # (1, )
        self.cx = nn.Parameter(torch.tensor(
            1.0, dtype=torch.float32), requires_grad=True)  # (1, )
        self.cy = nn.Parameter(torch.tensor(
            1.0, dtype=torch.float32), requires_grad=True)  # (1, )
        self.layer2 = nn.Linear(2, 2, bias=False)
        self.layer4 = nn.Linear(4, 4, bias=False)
        self.relu = nn.ReLU()

        self.noise_pct = torch.Tensor([noise_pct]).cuda()
        self.fxfy_noise = nn.Parameter(torch.ones(
            size=(1, 2), dtype=torch.float32), requires_grad=True)  # (N, 3)
        # self.layer2 = nn.Linear(2, 2, bias=False)
        # self.layer3 = nn.Linear(2, 2, bias=False)

    def forward(self, fxfy_gt):
        # order = 2, check our supplementary.

        PREDICT = "gt_ablation"
        fxfy_gt = torch.Tensor(fxfy_gt).cuda()

        if (PREDICT == "fxfy"):
            fxfy = torch.stack([self.fx**2 * self.W_temp,
                                self.fy**2 * self.W_temp,
                                self.W_temp,
                                self.H_temp])

        elif (PREDICT == "cxcy"):
            fxfy = torch.stack([self.W_temp,
                                self.W_temp,
                                self.cx**2 * self.W_temp,
                                self.cy**2 * self.W_temp])
        elif (PREDICT == "fxfycxcy"):
            fxfy = torch.stack([self.fx**2 * self.W_temp,
                                self.fy**2 * self.W_temp,
                                self.cx**2 * self.W_temp,
                                self.cy**2 * self.W_temp])

        elif (PREDICT == "fxfymodel"):
            x = torch.stack([self.H, self.W])
            fxfy = self.layer2(x)
            fxfy = self.relu(fxfy)
            fxfy = torch.stack([fxfy[0], -fxfy[1], self.W_temp, self.H_temp])
        elif (PREDICT == "cxcymodel"):
            x = torch.stack([self.H_temp, self.W_temp])
            fxfy = self.layer2(x)
            fxfy = self.relu(fxfy)
            fxfy = torch.stack([self.W_temp, self.H_temp, fxfy[0], fxfy[1]])
        elif (PREDICT == "fxfycxcymodel"):
            x = torch.stack([self.H, self.W, self.H_temp, self.W_temp])
            fxfy = self.layer4(x)
            fxfy = self.relu(fxfy)
            fxfy = torch.stack([fxfy[0], -fxfy[1], fxfy[2], fxfy[3]])
        elif (PREDICT == "gt_ablation"):
            # print(self.fxfy_noise)
            # print(fxfy_gt)
            # print(self.noise_pct)
            fxfy_noise = self.fxfy_noise*fxfy_gt[:2]*self.noise_pct
            fxfy_noise = torch.cat(
                [fxfy_noise, torch.Tensor([[0, 0]]).cuda()], -1)
            # fxfy_signs = torch.rand_like(fxfy_noise)
            # fxfy_signs[fxfy_signs] = 1
            # fxfy_signs[fxfy_signs > 0.5] = 1
            # fxfy_signs[fxfy_signs <= 0.5] = -1
            # fxfy_noise = fxfy_noise * fxfy_signs
            fxfy = (fxfy_gt + fxfy_noise)[0]

        # fxfy = torch.stack([self.fx**2 * self.W_temp,
        #                     self.fy**2 * self.W_temp,
        #                     self.cx**2 * self.W_temp,
        #                     self.cy**2 * self.W_temp])

        return fxfy
