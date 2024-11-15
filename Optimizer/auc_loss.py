import torch
import torch.nn.functional as F


class AUCMLoss(torch.nn.Module):
    def __init__(self, margin=1.0, imratio=None):
        super(AUCMLoss, self).__init__()
        self.margin = margin
        self.p = imratio
        # self.a = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32, device="cuda", requires_grad=True)).cuda()
        # self.b = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32, device="cuda", requires_grad=True)).cuda()
        # self.alpha = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32, device="cuda", requires_grad=True)).cuda()
        
        self.a = torch.zeros(1, dtype=torch.float32, device="cuda", requires_grad=True).cuda()
        self.b = torch.zeros(1, dtype=torch.float32, device="cuda", requires_grad=True).cuda()
        self.alpha = torch.zeros(1, dtype=torch.float32, device="cuda", requires_grad=True).cuda()
        # self.beta = torch.zeros(1, dtype=torch.float32, device="cuda", requires_grad=True).cuda()

    def forward(self, y_pred, y_true):
        # print(y_pred.shape)
        # print(y_true.shape)
        if self.p is None:
            self.p = (y_true == 1).float().sum() / y_true.shape[0]
        
        
        
        y_pred = y_pred.reshape(-1, 1)
        y_true = y_true.reshape(-1, 1)
        loss = (1 - self.p) * torch.mean((y_pred - self.a) ** 2 * (1 == y_true).float()) + \
               self.p * torch.mean((y_pred - self.b) ** 2 * (0 == y_true).float()) + \
               2 * self.alpha * (self.p * (1 - self.p) * self.margin + \
                                 torch.mean((self.p * y_pred * (0 == y_true).float() - (1 - self.p) * y_pred * (
                                             1 == y_true).float()))) - \
               self.p * (1 - self.p) * self.alpha ** 2
        
        
        # loss = (1 - self.p) * torch.mean((y_pred - self.a) ** 2 * (1 == y_true).float()) + \
        #        self.p * torch.mean((y_pred - self.b) ** 2 * (0 == y_true).float()) + \
        #        self.alpha * self.beta * self.p * (1 - self.p) * self.margin +\
        #        2 * self.alpha * (torch.mean(self.p * y_pred * (0 == y_true).float())) -\
        #        2 * self.beta * (torch.mean((1 - self.p) *  y_pred * (1 == y_true).float())) - \
        #        self.p * (1 - self.p) * self.alpha ** 2 -\
        #        self.p * (1 - self.p) * self.beta ** 2
        
        # loss = (1 - self.p) * torch.mean((y_pred - self.a) ** 2 * (1 == y_true).float()) + \
        #        self.p * torch.mean((y_pred - self.b) ** 2 * (0 == y_true).float()) 
        return loss

    

