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
        # self.layer2 = nn.Linear(2, 2, bias=False)
        # self.layer3 = nn.Linear(2, 2, bias=False)

    def forward(self):
        # order = 2, check our supplementary.

        PREDICT = "fxfycxcymodel"

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

        # predict fxfy
        # fxfy = torch.stack([self.fx**2 * self.W if PREDICT_FXFY else self.H,
        #                     self.fx**2 * self.W if PREDICT_FXFY else self.W,
        #                     self.H_temp if PREDICT_FXFY else self.fx ** 2 * self.H,
        #                     self.W_temp if PREDICT_FXFY else self.fy ** 2 * self.W,
        #                     # self.cx**2 * self.H_temp,
        #                     # self.cy**2 * self.W_temp
        #                     ])

        # Model test
        # x = torch.stack([self.H, self.W, self.H_temp, self.W_temp])
        # x = torch.stack([self.H, self.W])
        # fxfy = self.layer1(x)
        # print("self.fact: {}".format(self.fact))
        # fxfy = self.relu(fxfy)
        # fxfy = torch.stack([fxfy[0], -fxfy[1], self.W_temp, self.H_temp])
        fxfy = torch.stack([self.fx**2 * self.W_temp,
                            self.fy**2 * self.W_temp,
                            self.cx**2 * self.W_temp,
                            self.cy**2 * self.W_temp])

        # constant
        # x = torch.stack([self.H, self.W])
        # fxfy = torch.stack([self.fx,
        #                     self.fy,
        #                     self.H_temp,
        #                     self.W_temp])
        # fxfy = torch.stack([fxfy[0], fxfy[1], fxfy[2], fxfy[3]])

        return fxfy
