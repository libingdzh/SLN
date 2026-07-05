import torch
from ResNet_3D import ResNet18_3D_backbone
from Networks.Classification_Network_2D import Resnet18_backbone
from torch import nn
from Networks.Transformer_Network import Trans_cls_head_base
import ssl
from torchvision.models import resnet50, densenet121, vgg19, convnext_tiny
from Networks.onlyGCVIT import GCViT
ssl._create_default_https_context = ssl._create_unverified_context

# u
class MFNet_imgonly_new(nn.Module):
    def __init__(self, n_classes=2, no_cuda=False,drop=0):
        super(MFNet_imgonly_new, self).__init__()
        self.extractor_MR_DCE = ResNet18_3D_backbone(in_channels=3, no_cuda=no_cuda)
        self.extractor_MG_CC = Resnet18_backbone(in_channels=2)
        self.extractor_MG_MLO = Resnet18_backbone(in_channels=3)
        self.extractor_US_H = Resnet18_backbone(in_channels=2)
        self.extractor_US_V = Resnet18_backbone(in_channels=2)
        self.last_linear = nn.Linear(2560, n_classes)
        self.Trans = Trans_cls_head_base(d_model=512, num_classes=n_classes, nhead=4, num_layers=1,
                                         dim_feedforward=1024,
                                         dropout=drop, activation='relu')

    def forward(self, img_MR_DCE, img_MG_CC, img_MG_MLO, img_US_H, img_US_V):
        img_MR_DCE = self.extractor_MR_DCE(img_MR_DCE)
        img_MG_CC = self.extractor_MG_CC(img_MG_CC)
        img_MG_MLO = self.extractor_MG_MLO(img_MG_MLO)
        img_US_H = self.extractor_US_H(img_US_H)
        img_US_V = self.extractor_US_V(img_US_V)
        x = torch.cat((img_MR_DCE.unsqueeze(1),
                       img_MG_CC.unsqueeze(1), img_MG_MLO.unsqueeze(1),
                       img_US_H.unsqueeze(1), img_US_V.unsqueeze(1)), dim=1)
        x = self.Trans(x)
        return img_MR_DCE,img_MG_CC,img_MG_MLO,img_US_H,img_US_V, x

# u
class MFNet_imgonlyMG_new(nn.Module):
    def __init__(self, n_classes=2, no_cuda=False,drop=0):
        super(MFNet_imgonlyMG_new, self).__init__()
        self.extractor_MG_CC = Resnet18_backbone(in_channels=2)
        self.extractor_MG_MLO = Resnet18_backbone(in_channels=3)
        self.Trans = Trans_cls_head_base(d_model=512, num_classes=n_classes, nhead=4, num_layers=1,
                                         dim_feedforward=1024,
                                         dropout=drop, activation='relu')

    def forward(self, img_MG_CC, img_MG_MLO):
        img_MG_CC = self.extractor_MG_CC(img_MG_CC)
        img_MG_MLO = self.extractor_MG_MLO(img_MG_MLO)
        x = torch.cat((img_MG_CC.unsqueeze(1), img_MG_MLO.unsqueeze(1)), dim=1)
        x = self.Trans(x)
        return img_MG_CC,img_MG_MLO, x
# u
class MFNet_imgonlyMGUS_new(nn.Module):
    def __init__(self, n_classes=2, no_cuda=False,drop=0):
        super(MFNet_imgonlyMGUS_new, self).__init__()
        self.extractor_MG_CC = Resnet18_backbone(in_channels=2)
        self.extractor_MG_MLO = Resnet18_backbone(in_channels=3)
        self.extractor_US_H = Resnet18_backbone(in_channels=2)
        self.extractor_US_V = Resnet18_backbone(in_channels=2)
        self.last_linear = nn.Linear(2048, n_classes)
        self.Trans = Trans_cls_head_base(d_model=512, num_classes=n_classes, nhead=4, num_layers=1,
                                         dim_feedforward=1024,
                                         dropout=drop, activation='relu')

    def forward(self, img_MG_CC, img_MG_MLO, img_US_H, img_US_V):
        img_MG_CC = self.extractor_MG_CC(img_MG_CC)
        img_MG_MLO = self.extractor_MG_MLO(img_MG_MLO)
        img_US_H = self.extractor_US_H(img_US_H)
        img_US_V = self.extractor_US_V(img_US_V)
        x = torch.cat((img_MG_CC.unsqueeze(1), img_MG_MLO.unsqueeze(1),
                       img_US_H.unsqueeze(1), img_US_V.unsqueeze(1)), dim=1)
        x = self.Trans(x)
        return img_MG_CC,img_MG_MLO,img_US_H,img_US_V, x
# u
class MFNet_imgonlyUS_new(nn.Module):
    def __init__(self, n_classes=2, no_cuda=False,drop=0):
        super(MFNet_imgonlyUS_new, self).__init__()
        self.extractor_US_H = Resnet18_backbone(in_channels=2)
        self.extractor_US_V = Resnet18_backbone(in_channels=2)
        self.last_linear = nn.Linear(1024, n_classes)
        self.Trans = Trans_cls_head_base(d_model=512, num_classes=n_classes, nhead=4, num_layers=1,
                                         dim_feedforward=1024,
                                         dropout=drop, activation='relu')

    def forward(self, img_US_H, img_US_V):
        img_US_H = self.extractor_US_H(img_US_H)
        img_US_V = self.extractor_US_V(img_US_V)
        x = torch.cat((img_US_H.unsqueeze(1), img_US_V.unsqueeze(1)), dim=1)
        x = self.Trans(x)
        return img_US_H,img_US_V, x
# u

class MFNet_img_pathology_transcat_l3m_convnext(nn.Module):
    def __init__(self, n_classes=2, no_cuda=False,drop=0,clinical_inchannels=35,pathology_inchannels=20):
        super(MFNet_img_pathology_transcat_l3m_convnext, self).__init__()
        self.extractor_MR_DCE = ResNet18_3D_backbone(in_channels=3, no_cuda=no_cuda)
        self.extractor_MG_CC = Resnet18_backbone(in_channels=2)
        self.extractor_MG_MLO = Resnet18_backbone(in_channels=3)
        self.extractor_US_H = Resnet18_backbone(in_channels=2)
        self.extractor_US_V = Resnet18_backbone(in_channels=2)
        self.last_linear = nn.Linear(2560, n_classes)
        self.MLP_cln = nn.Sequential(
            nn.Linear(clinical_inchannels, 256, bias=False), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.Linear(256, 512, bias=True)
        )
        self.MLP_pathology = nn.Sequential(
            nn.Linear(pathology_inchannels, 256, bias=False), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.Linear(256, 512, bias=True)
        )
        self.channel_adjust = nn.Conv2d(5,3,kernel_size=1)
        self.Trans = convnext_tiny(pretrained=False)
        for m in self.Trans.features.modules():
            if isinstance(m, nn.Conv2d):
                m.stride = (1, 1)
        self.Trans.classifier = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                              nn.Flatten(),
                                              nn.Linear(768, 512))

        self.mlp_head = nn.Sequential(nn.LayerNorm(512 * 3), nn.Linear(512 * 3, n_classes))


    def forward(self, img_MR_DCE, img_MG_CC, img_MG_MLO, img_US_H, img_US_V, cln, pathology):
        clinical = self.MLP_cln(cln)
        pathgy = self.MLP_pathology(pathology)
        img_MR_DCE = self.extractor_MR_DCE(img_MR_DCE)
        img_MG_CC = self.extractor_MG_CC(img_MG_CC)
        img_MG_MLO = self.extractor_MG_MLO(img_MG_MLO)
        img_US_H = self.extractor_US_H(img_US_H)
        img_US_V = self.extractor_US_V(img_US_V)
        x = torch.cat((img_MR_DCE.unsqueeze(1),
                       img_MG_CC.unsqueeze(1), img_MG_MLO.unsqueeze(1),
                       img_US_H.unsqueeze(1), img_US_V.unsqueeze(1)), dim=1)
        x = x.view(x.size(0), x.size(1), 32, -1)
        x = self.channel_adjust(x)
        x = self.Trans(x)
        x = torch.cat((x, clinical, pathgy), dim=1)
        x = self.mlp_head(x)

        return img_MR_DCE,img_MG_CC,img_MG_MLO,img_US_H,img_US_V,clinical,pathgy,x
# u
class MFNet_img_pathology_transcat_l3m_orgTransformer(nn.Module):
    def __init__(self, n_classes=2, no_cuda=False,drop=0,clinical_inchannels=35,pathology_inchannels=20):
        super(MFNet_img_pathology_transcat_l3m_orgTransformer, self).__init__()
        self.extractor_MR_DCE = ResNet18_3D_backbone(in_channels=3, no_cuda=no_cuda)
        self.extractor_MG_CC = Resnet18_backbone(in_channels=2)
        self.extractor_MG_MLO = Resnet18_backbone(in_channels=3)
        self.extractor_US_H = Resnet18_backbone(in_channels=2)
        self.extractor_US_V = Resnet18_backbone(in_channels=2)
        self.last_linear = nn.Linear(2560, n_classes)
        self.MLP_cln = nn.Sequential(
            nn.Linear(clinical_inchannels, 256, bias=False), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.Linear(256, 512, bias=True)
        )
        self.MLP_pathology = nn.Sequential(
            nn.Linear(pathology_inchannels, 256, bias=False), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.Linear(256, 512, bias=True)
        )
        self.Trans = Trans_cls_head_base(d_model=512, num_classes=n_classes, nhead=4, num_layers=1,
                                         dim_feedforward=1024,
                                         dropout=drop, activation='relu')

    def forward(self, img_MR_DCE, img_MG_CC, img_MG_MLO, img_US_H, img_US_V, cln, pathology):
        clinical = self.MLP_cln(cln)
        pathgy = self.MLP_pathology(pathology)
        img_MR_DCE = self.extractor_MR_DCE(img_MR_DCE)
        img_MG_CC = self.extractor_MG_CC(img_MG_CC)
        img_MG_MLO = self.extractor_MG_MLO(img_MG_MLO)
        img_US_H = self.extractor_US_H(img_US_H)
        img_US_V = self.extractor_US_V(img_US_V)
        x = torch.cat((img_MR_DCE.unsqueeze(1),
                       img_MG_CC.unsqueeze(1), img_MG_MLO.unsqueeze(1),
                       img_US_H.unsqueeze(1), img_US_V.unsqueeze(1),
                       clinical.unsqueeze(1), pathgy.unsqueeze(1)), dim=1)
        x = self.Trans(x)
        return img_MR_DCE,img_MG_CC,img_MG_MLO,img_US_H,img_US_V,clinical,pathgy,x
# u
class MFNet_img_pathology_transcat_l3m_onlyGCViT(nn.Module):
    def __init__(self, n_classes=2, no_cuda=False,drop=0,clinical_inchannels=35,pathology_inchannels=20):
        super(MFNet_img_pathology_transcat_l3m_onlyGCViT, self).__init__()
        self.extractor_MR_DCE = ResNet18_3D_backbone(in_channels=3, no_cuda=no_cuda)
        self.extractor_MG_CC = Resnet18_backbone(in_channels=2)
        self.extractor_MG_MLO = Resnet18_backbone(in_channels=3)
        self.extractor_US_H = Resnet18_backbone(in_channels=2)
        self.extractor_US_V = Resnet18_backbone(in_channels=2)
        self.last_linear = nn.Linear(2560, n_classes)
        self.a_linear = nn.Linear(16, 32)
        self.MLP_cln = nn.Sequential(
            nn.Linear(clinical_inchannels, 256, bias=False), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.Linear(256, 512, bias=True)
        )
        self.MLP_pathology = nn.Sequential(
            nn.Linear(pathology_inchannels, 256, bias=False), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.Linear(256, 512, bias=True)
        )
        self.Trans = GCViT()

        self.mlp_head = nn.Sequential(nn.LayerNorm(512 * 3), nn.Linear(512 * 3, n_classes))


    def forward(self, img_MR_DCE, img_MG_CC, img_MG_MLO, img_US_H, img_US_V, cln, pathology):
        clinical = self.MLP_cln(cln)
        pathgy = self.MLP_pathology(pathology)
        img_MR_DCE = self.extractor_MR_DCE(img_MR_DCE)
        img_MG_CC = self.extractor_MG_CC(img_MG_CC)
        img_MG_MLO = self.extractor_MG_MLO(img_MG_MLO)
        img_US_H = self.extractor_US_H(img_US_H)
        img_US_V = self.extractor_US_V(img_US_V)
        x = torch.cat((img_MR_DCE.unsqueeze(1),
                       img_MG_CC.unsqueeze(1), img_MG_MLO.unsqueeze(1),
                       img_US_H.unsqueeze(1), img_US_V.unsqueeze(1)), dim=1)
        x = x.view(x.size(0), x.size(1), 32, -1)
        x = self.a_linear(x)
        x = self.Trans(x)
        x = torch.cat((x, clinical, pathgy), dim=1)
        x = self.mlp_head(x)

        return img_MR_DCE,img_MG_CC,img_MG_MLO,img_US_H,img_US_V,clinical,pathgy,x
# u
class MFNet_img_pathology_transcat_l3m_resnet50(nn.Module):
    def __init__(self, n_classes=2, no_cuda=False,drop=0,clinical_inchannels=35,pathology_inchannels=20):
        super(MFNet_img_pathology_transcat_l3m_resnet50, self).__init__()
        self.extractor_MR_DCE = ResNet18_3D_backbone(in_channels=3, no_cuda=no_cuda)
        self.extractor_MG_CC = Resnet18_backbone(in_channels=2)
        self.extractor_MG_MLO = Resnet18_backbone(in_channels=3)
        self.extractor_US_H = Resnet18_backbone(in_channels=2)
        self.extractor_US_V = Resnet18_backbone(in_channels=2)
        self.last_linear = nn.Linear(2560, n_classes)
        self.MLP_cln = nn.Sequential(
            nn.Linear(clinical_inchannels, 256, bias=False), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.Linear(256, 512, bias=True)
        )
        self.MLP_pathology = nn.Sequential(
            nn.Linear(pathology_inchannels, 256, bias=False), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.Linear(256, 512, bias=True)
        )
        self.Trans = resnet50(pretrained=False)
        self.Trans.conv1 = nn.Conv2d(5,64,kernel_size=7, stride=1, padding=3, bias=False)
        self.Trans.maxpool = nn.Identity()
        self.Trans.fc = nn.Linear(2048, 512)

        self.mlp_head = nn.Sequential(nn.LayerNorm(512 * 3), nn.Linear(512 * 3, n_classes))


    def forward(self, img_MR_DCE, img_MG_CC, img_MG_MLO, img_US_H, img_US_V, cln, pathology):
        clinical = self.MLP_cln(cln)
        pathgy = self.MLP_pathology(pathology)
        img_MR_DCE = self.extractor_MR_DCE(img_MR_DCE)
        img_MG_CC = self.extractor_MG_CC(img_MG_CC)
        img_MG_MLO = self.extractor_MG_MLO(img_MG_MLO)
        img_US_H = self.extractor_US_H(img_US_H)
        img_US_V = self.extractor_US_V(img_US_V)
        x = torch.cat((img_MR_DCE.unsqueeze(1),
                       img_MG_CC.unsqueeze(1), img_MG_MLO.unsqueeze(1),
                       img_US_H.unsqueeze(1), img_US_V.unsqueeze(1)), dim=1)
        x = x.view(x.size(0), x.size(1), 32, -1)
        x = self.Trans(x)
        x = torch.cat((x, clinical, pathgy), dim=1)
        x = self.mlp_head(x)

        return img_MR_DCE,img_MG_CC,img_MG_MLO,img_US_H,img_US_V,clinical,pathgy,x


# u
class MFNet_img_pathology_transcat_l3m_densenet121(nn.Module):
    def __init__(self, n_classes=2, no_cuda=False,drop=0,clinical_inchannels=35,pathology_inchannels=20):
        super(MFNet_img_pathology_transcat_l3m_densenet121, self).__init__()
        self.extractor_MR_DCE = ResNet18_3D_backbone(in_channels=3, no_cuda=no_cuda)
        self.extractor_MG_CC = Resnet18_backbone(in_channels=2)
        self.extractor_MG_MLO = Resnet18_backbone(in_channels=3)
        self.extractor_US_H = Resnet18_backbone(in_channels=2)
        self.extractor_US_V = Resnet18_backbone(in_channels=2)
        self.last_linear = nn.Linear(2560, n_classes)
        self.MLP_cln = nn.Sequential(
            nn.Linear(clinical_inchannels, 256, bias=False), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.Linear(256, 512, bias=True)
        )
        self.MLP_pathology = nn.Sequential(
            nn.Linear(pathology_inchannels, 256, bias=False), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.Linear(256, 512, bias=True)
        )
        self.Trans = densenet121(pretrained=False)
        self.Trans.features[0] = nn.Conv2d(5,64,kernel_size=7, stride=1, padding=3, bias=False)
        self.Trans.features[1] = nn.Identity()
        self.Trans.classifier = nn.Linear(1024, 512)

        self.mlp_head = nn.Sequential(nn.LayerNorm(512 * 3), nn.Linear(512 * 3, n_classes))


    def forward(self, img_MR_DCE, img_MG_CC, img_MG_MLO, img_US_H, img_US_V, cln, pathology):
        clinical = self.MLP_cln(cln)
        pathgy = self.MLP_pathology(pathology)
        img_MR_DCE = self.extractor_MR_DCE(img_MR_DCE)
        img_MG_CC = self.extractor_MG_CC(img_MG_CC)
        img_MG_MLO = self.extractor_MG_MLO(img_MG_MLO)
        img_US_H = self.extractor_US_H(img_US_H)
        img_US_V = self.extractor_US_V(img_US_V)
        x = torch.cat((img_MR_DCE.unsqueeze(1),
                       img_MG_CC.unsqueeze(1), img_MG_MLO.unsqueeze(1),
                       img_US_H.unsqueeze(1), img_US_V.unsqueeze(1)), dim=1)
        x = x.view(x.size(0), x.size(1), 32, -1)
        x = self.Trans(x)
        x = torch.cat((x, clinical, pathgy), dim=1)
        x = self.mlp_head(x)

        return img_MR_DCE,img_MG_CC,img_MG_MLO,img_US_H,img_US_V,clinical,pathgy,x

# u
class MFNet_img_pathology_transcat_l3m_vgg19(nn.Module):
    def __init__(self, n_classes=2, no_cuda=False,drop=0,clinical_inchannels=35,pathology_inchannels=20):
        super(MFNet_img_pathology_transcat_l3m_vgg19, self).__init__()
        self.extractor_MR_DCE = ResNet18_3D_backbone(in_channels=3, no_cuda=no_cuda)
        self.extractor_MG_CC = Resnet18_backbone(in_channels=2)
        self.extractor_MG_MLO = Resnet18_backbone(in_channels=3)
        self.extractor_US_H = Resnet18_backbone(in_channels=2)
        self.extractor_US_V = Resnet18_backbone(in_channels=2)
        self.last_linear = nn.Linear(2560, n_classes)
        self.MLP_cln = nn.Sequential(
            nn.Linear(clinical_inchannels, 256, bias=False), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.Linear(256, 512, bias=True)
        )
        self.MLP_pathology = nn.Sequential(
            nn.Linear(pathology_inchannels, 256, bias=False), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.Linear(256, 512, bias=True)
        )
        vgg = vgg19(pretrained=False).features
        first_conv = nn.Conv2d(5,64, kernel_size=3, padding=1)
        self.Trans = nn.Sequential(first_conv, *list(vgg.children())[1:-1], nn.AdaptiveAvgPool2d(1))

        self.mlp_head = nn.Sequential(nn.LayerNorm(512 * 3), nn.Linear(512 * 3, n_classes))


    def forward(self, img_MR_DCE, img_MG_CC, img_MG_MLO, img_US_H, img_US_V, cln, pathology):
        clinical = self.MLP_cln(cln)
        pathgy = self.MLP_pathology(pathology)
        img_MR_DCE = self.extractor_MR_DCE(img_MR_DCE)
        img_MG_CC = self.extractor_MG_CC(img_MG_CC)
        img_MG_MLO = self.extractor_MG_MLO(img_MG_MLO)
        img_US_H = self.extractor_US_H(img_US_H)
        img_US_V = self.extractor_US_V(img_US_V)
        x = torch.cat((img_MR_DCE.unsqueeze(1),
                       img_MG_CC.unsqueeze(1), img_MG_MLO.unsqueeze(1),
                       img_US_H.unsqueeze(1), img_US_V.unsqueeze(1)), dim=1)
        x = x.view(x.size(0), x.size(1), 16, -1)
        x = self.Trans(x)
        x = x.squeeze(dim=2)
        x = x.squeeze(dim=2)
        x = torch.cat((x, clinical, pathgy), dim=1)
        x = self.mlp_head(x)

        return img_MR_DCE,img_MG_CC,img_MG_MLO,img_US_H,img_US_V,clinical,pathgy,x

# u
class MFNet_imgonlyMRMG_new(nn.Module):
    def __init__(self, n_classes=2, no_cuda=False,drop=0):
        super(MFNet_imgonlyMRMG_new, self).__init__()
        self.extractor_MR_DCE = ResNet18_3D_backbone(in_channels=3, no_cuda=no_cuda)
        self.extractor_MG_CC = Resnet18_backbone(in_channels=2)
        self.extractor_MG_MLO = Resnet18_backbone(in_channels=3)
        self.Trans = Trans_cls_head_base(d_model=512, num_classes=n_classes, nhead=4, num_layers=1,
                                         dim_feedforward=1024,
                                         dropout=drop, activation='relu')

    def forward(self, img_MR_DCE, img_MG_CC, img_MG_MLO):
        img_MR_DCE = self.extractor_MR_DCE(img_MR_DCE)
        img_MG_CC = self.extractor_MG_CC(img_MG_CC)
        img_MG_MLO = self.extractor_MG_MLO(img_MG_MLO)
        x = torch.cat((img_MR_DCE.unsqueeze(1),
                       img_MG_CC.unsqueeze(1), img_MG_MLO.unsqueeze(1)), dim=1)
        x = self.Trans(x)
        return img_MR_DCE,img_MG_CC,img_MG_MLO, x

# u
class MFNet_imgonlyMRUS_new(nn.Module):
    def __init__(self, n_classes=2, no_cuda=False,drop=0):
        super(MFNet_imgonlyMRUS_new, self).__init__()
        self.extractor_MR_DCE = ResNet18_3D_backbone(in_channels=3, no_cuda=no_cuda)
        self.extractor_US_H = Resnet18_backbone(in_channels=2)
        self.extractor_US_V = Resnet18_backbone(in_channels=2)
        self.last_linear = nn.Linear(1536, n_classes)
        self.Trans = Trans_cls_head_base(d_model=512, num_classes=n_classes, nhead=4, num_layers=1,
                                         dim_feedforward=1024,
                                         dropout=drop, activation='relu')

    def forward(self, img_MR_DCE, img_US_H, img_US_V):
        img_MR_DCE = self.extractor_MR_DCE(img_MR_DCE)
        img_US_H = self.extractor_US_H(img_US_H)
        img_US_V = self.extractor_US_V(img_US_V)
        x = torch.cat((img_MR_DCE.unsqueeze(1),
                       img_US_H.unsqueeze(1), img_US_V.unsqueeze(1)), dim=1)
        x = self.Trans(x)
        return img_MR_DCE,img_US_H,img_US_V, x


# u
class MFNet_imgonlyMR_new(nn.Module):
    def __init__(self, n_classes=2, no_cuda=False,drop=0):
        super(MFNet_imgonlyMR_new, self).__init__()
        self.extractor_MR_DCE = ResNet18_3D_backbone(in_channels=3, no_cuda=no_cuda)
        self.Trans = Trans_cls_head_base(d_model=512, num_classes=n_classes, nhead=4, num_layers=1,
                                         dim_feedforward=1024,
                                         dropout=drop, activation='relu')

    def forward(self, img_MR_DCE):
        img_MR_DCE = self.extractor_MR_DCE(img_MR_DCE)
        x = img_MR_DCE.unsqueeze(1)
        x = self.Trans(x)
        return img_MR_DCE, x

# u
class MFNet_img_onlypathology_transcat_l3m(nn.Module):
    def __init__(self, n_classes=2, no_cuda=False,drop=0,clinical_inchannels=35,pathology_inchannels=20):
        super(MFNet_img_onlypathology_transcat_l3m, self).__init__()
        self.MLP_cln = nn.Sequential(
            nn.Linear(clinical_inchannels, 256, bias=False), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.Linear(256, 512, bias=True)
        )
        self.MLP_pathology = nn.Sequential(
            nn.Linear(pathology_inchannels, 256, bias=False), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.Linear(256, 512, bias=True)
        )
        self.Trans = Trans_cls_head_base(d_model=512, num_classes=n_classes, nhead=4, num_layers=1,
                                         dim_feedforward=1024,
                                         dropout=drop, activation='relu')

    def forward(self, cln, pathology):
        clinical = self.MLP_cln(cln)
        pathgy = self.MLP_pathology(pathology)
        x = torch.cat((clinical.unsqueeze(1), pathgy.unsqueeze(1)), dim=1)
        x = self.Trans(x)
        return clinical,pathgy,x
# u
class MFNet_img_onlypathologynotdistance_transcat_l3m(nn.Module):
    def __init__(self, n_classes=2, no_cuda=False,drop=0,clinical_inchannels=34,pathology_inchannels=20):
        super(MFNet_img_onlypathologynotdistance_transcat_l3m, self).__init__()
        self.MLP_cln = nn.Sequential(
            nn.Linear(clinical_inchannels, 256, bias=True), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.Linear(256, 512, bias=True)
        )
        self.MLP_pathology = nn.Sequential(
            nn.Linear(pathology_inchannels, 256, bias=True), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.Linear(256, 512, bias=True)
        )
        self.Trans = Trans_cls_head_base(d_model=512, num_classes=n_classes, nhead=4, num_layers=1,
                                         dim_feedforward=1024,
                                         dropout=drop, activation='relu')

    def forward(self, cln, pathology):
        clinical = self.MLP_cln(cln)
        pathgy = self.MLP_pathology(pathology)
        x = torch.cat((clinical.unsqueeze(1), pathgy.unsqueeze(1)), dim=1)
        x = self.Trans(x)
        return clinical,pathgy,x
# u
class MFNet_img_pathology_transcat_l3m_orgTransformer_1(nn.Module):
    def __init__(self, n_classes=2, no_cuda=False,drop=0,clinical_inchannels=35,pathology_inchannels=20):
        super(MFNet_img_pathology_transcat_l3m_orgTransformer_1, self).__init__()
        self.extractor_MR_DCE = ResNet18_3D_backbone(in_channels=3, no_cuda=no_cuda)
        self.extractor_MG_CC = Resnet18_backbone(in_channels=2)
        self.extractor_MG_MLO = Resnet18_backbone(in_channels=3)
        self.extractor_US_H = Resnet18_backbone(in_channels=2)
        self.extractor_US_V = Resnet18_backbone(in_channels=2)
        self.last_linear = nn.Linear(2560, n_classes)
        self.MLP_cln = nn.Sequential(
            nn.Linear(clinical_inchannels, 256, bias=False), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.Linear(256, 512, bias=False)
        )
        self.MLP_pathology = nn.Sequential(
            nn.Linear(pathology_inchannels, 256, bias=False), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.Linear(256, 512, bias=False)
        )
        self.Trans = Trans_cls_head_base(d_model=512, num_classes=n_classes, nhead=4, num_layers=1,
                                         dim_feedforward=1024,
                                         dropout=drop, activation='relu')

    def forward(self, img_MR_DCE, img_MG_CC, img_MG_MLO, img_US_H, img_US_V, cln, pathology):
        clinical = self.MLP_cln(cln)
        pathgy = self.MLP_pathology(pathology)
        img_MR_DCE = self.extractor_MR_DCE(img_MR_DCE)
        img_MG_CC = self.extractor_MG_CC(img_MG_CC)
        img_MG_MLO = self.extractor_MG_MLO(img_MG_MLO)
        img_US_H = self.extractor_US_H(img_US_H)
        img_US_V = self.extractor_US_V(img_US_V)
        x = torch.cat((img_MR_DCE.unsqueeze(1),
                       img_MG_CC.unsqueeze(1), img_MG_MLO.unsqueeze(1),
                       img_US_H.unsqueeze(1), img_US_V.unsqueeze(1),
                       clinical.unsqueeze(1), pathgy.unsqueeze(1)), dim=1)
        x = self.Trans(x)
        return img_MR_DCE,img_MG_CC,img_MG_MLO,img_US_H,img_US_V,clinical,pathgy,x
# u
class MFNet_img_pathology_transcat_l3m_orgTransformer_1_notdistance(nn.Module):
    def __init__(self, n_classes=2, no_cuda=False,drop=0,clinical_inchannels=34,pathology_inchannels=20):
        super(MFNet_img_pathology_transcat_l3m_orgTransformer_1_notdistance, self).__init__()
        self.extractor_MR_DCE = ResNet18_3D_backbone(in_channels=3, no_cuda=no_cuda)
        self.extractor_MG_CC = Resnet18_backbone(in_channels=2)
        self.extractor_MG_MLO = Resnet18_backbone(in_channels=3)
        self.extractor_US_H = Resnet18_backbone(in_channels=2)
        self.extractor_US_V = Resnet18_backbone(in_channels=2)
        self.last_linear = nn.Linear(2560, n_classes)
        self.MLP_cln = nn.Sequential(
            nn.Linear(clinical_inchannels, 256, bias=False), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.Linear(256, 512, bias=True)
        )
        self.MLP_pathology = nn.Sequential(
            nn.Linear(pathology_inchannels, 256, bias=False), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.Linear(256, 512, bias=True)
        )
        self.Trans = Trans_cls_head_base(d_model=512, num_classes=n_classes, nhead=4, num_layers=1,
                                         dim_feedforward=1024,
                                         dropout=drop, activation='relu')

    def forward(self, img_MR_DCE, img_MG_CC, img_MG_MLO, img_US_H, img_US_V, cln, pathology):
        clinical = self.MLP_cln(cln)
        pathgy = self.MLP_pathology(pathology)
        img_MR_DCE = self.extractor_MR_DCE(img_MR_DCE)
        img_MG_CC = self.extractor_MG_CC(img_MG_CC)
        img_MG_MLO = self.extractor_MG_MLO(img_MG_MLO)
        img_US_H = self.extractor_US_H(img_US_H)
        img_US_V = self.extractor_US_V(img_US_V)
        x = torch.cat((img_MR_DCE.unsqueeze(1),
                       img_MG_CC.unsqueeze(1), img_MG_MLO.unsqueeze(1),
                       img_US_H.unsqueeze(1), img_US_V.unsqueeze(1),
                       clinical.unsqueeze(1), pathgy.unsqueeze(1)), dim=1)
        x = self.Trans(x)
        return img_MR_DCE,img_MG_CC,img_MG_MLO,img_US_H,img_US_V,clinical,pathgy,x

if __name__ == '__main__':
    print('-------------------')
