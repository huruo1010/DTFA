import torch.nn as nn
import torch.nn.functional as F
import torch

class MgdLoss(nn.Module):
    def __init__(self,
                 student_channels,
                 teacher_channels,
                 alpha_mgd=0.000001,
                 lambda_mgd=0.65,
                 ):
        super(MgdLoss, self).__init__()
        self.alpha_mgd = alpha_mgd
        self.lambda_mgd = lambda_mgd

        if student_channels != teacher_channels:
            self.align = nn.Conv2d(student_channels, teacher_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.align = None

        self.generation = nn.Sequential(
            nn.Conv2d(teacher_channels, teacher_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(teacher_channels, teacher_channels, kernel_size=3, padding=1))

    def forward(self,
                preds_S,
                preds_T):
        assert preds_S.shape[-2:] == preds_T.shape[-2:]

        if self.align is not None:
            preds_S = self.align(preds_S)

        loss = self.get_dis_loss(preds_S, preds_T) * self.alpha_mgd

        return loss

    def get_dis_loss(self, preds_S, preds_T):
        loss_mse = nn.MSELoss(reduction='sum')
        N, C, H, W = preds_T.shape

        device = preds_S.device
        mat = torch.rand((N, 1, H, W)).to(device)
        mat = torch.where(mat > 1 - self.lambda_mgd, 0, 1).to(device)

        masked_fea = torch.mul(preds_S, mat)
        new_fea = self.generation(masked_fea)

        dis_loss = loss_mse(new_fea, preds_T)

        return dis_loss


class ShallowPym(nn.Module):
    def __init__(self):
        super(ShallowPym, self).__init__()
        self.tb_conv3 = nn.Conv2d(128, 128, (1, 1))
        self.tb_conv4 = nn.Conv2d(64, 128, (1, 1))

    def _upsample_add(self, x, y):
        _, _, H, W = y.shape
        return nn.functional.interpolate(x, size=(H, W), mode='bilinear') + y

    def from_top_to_bottom(self,feat3, feat4):
        feat3 = self.tb_conv3(feat3)
        feat4 = self.tb_conv4(feat4)

        feat4 = self._upsample_add(feat3, feat4)

        return feat3, feat4

    def forward(self, x3, x4):
        feat3, feat4 = self.from_top_to_bottom(x3, x4)

        return feat3, feat4

class DeepPym(nn.Module):
    def __init__(self):
        super(DeepPym, self).__init__()
        self.tb_conv3 = nn.Conv2d(512, 512, (1, 1))
        self.tb_conv4 = nn.Conv2d(256, 512, (1, 1))

    ###自上而下的上采样模块
    def _upsample_add(self, x, y):
        _, _, H, W = y.shape
        return nn.functional.interpolate(x, size=(H, W), mode='bilinear') + y

    def from_top_to_bottom(self,feat3, feat4):
        feat3 = self.tb_conv3(feat3)
        feat4 = self.tb_conv4(feat4)

        feat4 = self._upsample_add(feat3, feat4)

        return feat3, feat4

    def forward(self, x3, x4):
        feat3, feat4 = self.from_top_to_bottom(x3, x4)

        return feat3, feat4

class AFBLoss(nn.Module):
    def __init__(self,
                 alpha_mgd=0.000001,
                 lambda_mgd=0.65,
                 ):
        super(AFBLoss, self).__init__()
        self.ShallowPym = ShallowPym()
        self.loss = MgdLoss(128, 128, alpha_mgd, lambda_mgd)

    def forward(self, preds_S, preds_T):
        S_feats = self.ShallowPym(preds_S[0],preds_S[1])
        T_feats = self.ShallowPym(preds_T[0],preds_T[1])
        loss = self.loss(S_feats[1], T_feats[1])
        return loss

if __name__ == '__main__':
    x1 = torch.randn(8, 512, 20, 20)
    x2 = torch.randn(8, 256, 40, 40)
    x3 = torch.randn(8, 128, 80, 80)
    x4 = torch.randn(8, 64, 160, 160)
    S_feats = (x3,x4)
    T_feats = S_feats

    net = ShallowPymLoss()
    loss = net(S_feats,T_feats)
    print(loss)
    net = DeepPymLoss()
    S_feats = (x1, x2)
    T_feats = S_feats
    loss = net(S_feats, T_feats)
    print(loss)