import torch
from torch import nn
from leaf_pytorch import get_frontend
from models.model_helper import get_classifier


# class Classifier(nn.Module):
#     def __init__(self, cfg):
#         super(Classifier, self).__init__()
#         self.cfg = cfg
#         self.features = get_frontend(cfg)
#         self.model = get_classifier(cfg['model'])
#
#     def forward(self, x):
#         out = self.features(x)
#         out = out.unsqueeze(1)
#         print('out :',out.shape)
#         out = self.model(out)
#         return out


class Classifier(nn.Module):
    def __init__(self, cfg):
        super(Classifier, self).__init__()
        self.cfg = cfg
        self.features = get_frontend(cfg)
        self.model = get_classifier(cfg['model'])
        self.logsoftmax=nn.LogSoftmax(dim=-1)


    def forward(self, x):
        # print(x.shape)
        if(len(x.shape))==3:
            x=x.unsqueeze(2)
        out=torch.permute(x,(1,0,2,3))
        out=torch.cat([(self.features(x)).unsqueeze(1) for x in out],1)
        # print(out.shape)
        # out =out.unsqueeze(0)
        # out = out.unsqueeze(1)
        out = self.model(out)
        out=self.logsoftmax(out)

        return out
