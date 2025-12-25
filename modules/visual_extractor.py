import torch
import torch.nn as nn
import torchvision.models as models
from modules.Swing_Transformer import SwinTransformer as STBackbone

class VisualExtractor(nn.Module):
    def __init__(self, args):
        super(VisualExtractor, self).__init__()
        self.CNN = STBackbone(
                    img_size=224, 
                    embed_dim=192,
                    depths=[2, 2, 18, 2],
                    num_heads=[6, 12, 24, 48],
                    window_size=7,
                    num_classes=1000
                    )
        
        self.CNN.load_weights('./Swin/large.pth')

    def forward(self, images):
        patch_feats = self.CNN(images)  
        avg_feats = torch.mean(patch_feats,dim=1)
        return patch_feats, avg_feats
