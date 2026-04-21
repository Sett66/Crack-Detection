import torch
from torch import nn
from mmcls.SAVSS_dev.models.SAVSS.layer import SAVSS_Backbone
from models.EgdeEnhancedUDecoder import EnhancedUNetDecoder


from CompareModels.Unet import UNet
from CompareModels.DeepCrack import DeepCrackNet
from CompareModels.FPHBN import FPHBNNet
from CompareModels.CrackFormer import crackformer
from CompareModels.CMUNeXt import CMUNeXt
from CompareModels.PAF_Net import PAF
from CompareModels.CTCrackSeg import TransMUNet       #   CTCrackSeg
from CompareModels.DconnNet.DconnNet import DconnNet
from CompareModels.DSC_Net.DSCNet import DSCNet
from CompareModels.Crackmer.Crackmer import Crackmer
from CompareModels.SimCrack import SegPMIUNet         # SimCrack
from CompareModels.vmunet.vmunet import VMUNet
from CompareModels.mambavision import MM_mamba_vision


class Decoder(nn.Module):
    def __init__(self, backbone, args=None):
        super().__init__()
        self.args = args
        self.backbone = backbone
        # self.head = EnhancedUNetDecoder()

    def forward(self, samples):
        outs = self.backbone(samples)
        # outs = self.head(outs)

        outs = outs[5]

        return outs

class DiceLoss(nn.Module):
    def __init__(self, smooth=1., dims=(-2, -1)):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.dims = dims

    def forward(self, x, y):
        tp = (x * y).sum(self.dims)
        fp = (x * (1 - y)).sum(self.dims)
        fn = ((1 - x) * y).sum(self.dims)
        dc = (2 * tp + self.smooth) / (2 * tp + fp + fn + self.smooth)
        dc = dc.mean()

        return 1 - dc

class bce_dice(nn.Module):
    def __init__(self, args):
        super(bce_dice, self).__init__()
        self.bce_fn = nn.BCEWithLogitsLoss()
        self.dice_fn = DiceLoss()
        self.args = args

    def forward(self, y_pred, y_true):
        bce = self.bce_fn(y_pred, y_true)
        dice = self.dice_fn(y_pred.sigmoid(), y_true)
        return self.args.BCELoss_ratio * bce + self.args.DiceLoss_ratio * dice



def build(args):
    device = torch.device(args.device)
    args.device = torch.device(args.device)

    # backbone = TransMUNet() # CTCrackSeg
    # backbone = DSCNet(n_channels=3, n_classes=1, kernel_size=9, extend_scope=1.0, if_offset=True, number=32, dim=1,
    #                 device=args.device)
    # backbone = DconnNet()

    # backbone = UNet(3, 1, 64)
    # backbone = DeepCrackNet(3, 1, 64)
    # backbone = FPHBNNet(3, 1, 64)
    backbone = crackformer()
    # backbone = SegPMIUNet(3, 1)
    # backbone = VSSM(3, 1)
    # backbone = Crackmer()
    # backbone = MFDB(3,1)
    # backbone = CMUNeXt()
    # backbone = PAF(3,1,64,'batch')
    # backbone = VMUNet()
    # backbone.load_from()


    # backbone = SAVSS_Backbone(in_ch=3, base_dim=16, depths=[1,1,1,1], mamba_cfg={'d_state':16,'expand':2,'conv_size':7,'dt_init':'random'})

    model = Decoder(backbone, args)

    criterion = bce_dice(args)
    criterion.to(device)

    return model, criterion