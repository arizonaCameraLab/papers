import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
import torchvision.transforms.functional as TF

#####################################
# 1. ResidualBlock and SEBlock
#####################################

class ResidualBlock(nn.Module):
    """
    Conv+BN+ReLU，+ skip
    """
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)
                                                                                                                                                                                                                                                 
class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation 
    """
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

#####################################
# 2. UnetSkipConnectionBlock，with use_attention and use_residual
#####################################
class UnetSkipConnectionBlock(nn.Module):

    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None,
                 outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, 
                 use_dropout=False, use_attention=False, use_residual=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        self.use_attention = use_attention
        self.use_residual = use_residual
        use_bias = False

        if input_nc is None:
            input_nc = outer_nc


        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)

        def up_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=use_bias),
                norm_layer(out_channels),
                nn.ReLU(True)
            )

        if outermost:
            upconv = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(inner_nc * 2, outer_nc, kernel_size=3, padding=1, bias=use_bias),
                nn.ReLU(True)
            )
            down = [downconv]
            model = down + ([submodule] if submodule is not None else []) + [upconv]
        elif innermost:
            upconv = up_block(inner_nc, outer_nc)
            down = [downrelu, downconv]
            model = down + [upconv]
        else:

            upconv = up_block(inner_nc * 2, outer_nc)
            down = [downrelu, downconv, downnorm]
            if submodule is None:
                model = down + [upconv]
            else:
                model = down + [submodule] + [upconv]
                if use_dropout:
                    model += [nn.Dropout(0.5)]
        self.model = nn.Sequential(*model)

        if not outermost and self.use_attention:
            self.attention = SEBlock(2 * outer_nc)
        else:
            self.attention = None
        
        if not outermost and self.use_residual:
            self.res_block = ResidualBlock(2 * outer_nc)
        else:
            self.res_block = None

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            out = self.model(x)
            x_cat = torch.cat([x, out], 1)
            if self.attention is not None:
                x_cat = self.attention(x_cat)
            if self.res_block is not None:
                x_cat = self.res_block(x_cat)
            return x_cat

#####################################
#####################################
class UNet(nn.Module):

    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False,
                 use_attention=True, use_residual=True):
        super(UNet, self).__init__()
        
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None,
                                             submodule=None, innermost=True, norm_layer=norm_layer)
        for _ in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None,
                                                 submodule=unet_block, norm_layer=norm_layer, 
                                                 use_dropout=use_dropout,
                                                 use_attention=use_attention,
                                                 use_residual=use_residual)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None,
                                             submodule=unet_block, norm_layer=norm_layer,
                                             use_attention=use_attention,
                                             use_residual=use_residual)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None,
                                             submodule=unet_block, norm_layer=norm_layer,
                                             use_attention=use_attention,
                                             use_residual=use_residual)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None,
                                             submodule=unet_block, norm_layer=norm_layer,
                                             use_attention=use_attention,
                                             use_residual=use_residual)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc,
                                             submodule=unet_block, outermost=True, norm_layer=norm_layer)
        self.model = unet_block

    def forward(self, x):
        return self.model(x)
    
    
class HeightReflectanceNet(nn.Module):
    def __init__(self, in_channels=1, base_channels=64, H=128, W=128, height_scale=100.0):
        super(HeightReflectanceNet, self).__init__()
        
        self.height_scale = height_scale
        
        self.shared_net = UNet(input_nc=in_channels, output_nc=base_channels, num_downs=5, use_dropout=False)
        
        self.height_decoder = nn.Sequential(
            nn.Conv2d(base_channels, base_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels // 2),
            nn.ReLU(inplace=True),
            ResidualBlock(base_channels // 2),
            nn.Conv2d(base_channels // 2, base_channels // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels // 4),
            nn.ReLU(inplace=True),
            ResidualBlock(base_channels // 4),
            nn.Conv2d(base_channels // 4, 1, kernel_size=3, padding=1)
        )
        
        self.reflectance_decoder = nn.Sequential(
            nn.Conv2d(base_channels, base_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels // 2),
            nn.ReLU(inplace=True),
            ResidualBlock(base_channels // 2),
            nn.Conv2d(base_channels // 2, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, high_freq):
        features = self.shared_net(high_freq)
        
        height_logits = self.height_decoder(features)
        reflectance = self.reflectance_decoder(features)
        
        height = self.height_scale * torch.sigmoid(height_logits)
        height = height * (reflectance > 0.01).float()
        return height, reflectance


class MultiChannelFusionNet(nn.Module):
    def __init__(self, in_channels=3, base_channels=64, use_attention=True, use_residual=True, height_scale=100.0):
        super(MultiChannelFusionNet, self).__init__()
        assert in_channels == 3, "RGB"

        self.height_scale = height_scale

        num_down = 3
        self.unet_R = UNet(input_nc=1, output_nc=base_channels, num_downs=5,
                           use_attention=use_attention, use_residual=use_residual)
        self.unet_G = UNet(input_nc=1, output_nc=base_channels, num_downs=5,
                           use_attention=use_attention, use_residual=use_residual)
        self.unet_B = UNet(input_nc=1, output_nc=base_channels, num_downs=5,
                           use_attention=use_attention, use_residual=use_residual)

        # ===== fusion net =====
        self.fusion_net = UNet(input_nc=3 * base_channels,
                               output_nc=base_channels,
                               num_downs=4,
                               use_attention=use_attention,
                               use_residual=use_residual)

        # ===== height branch =====
        self.height_decoder = nn.Sequential(
            nn.Conv2d(base_channels, base_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels // 2),
            nn.ReLU(inplace=True),
            ResidualBlock(base_channels // 2),
            nn.Conv2d(base_channels // 2, base_channels // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels // 4),
            nn.ReLU(inplace=True),
            ResidualBlock(base_channels // 4),
            nn.Conv2d(base_channels // 4, 1, kernel_size=3, padding=1)
        )

        # ===== reflectance branch =====
        self.reflectance_decoder = nn.Sequential(
            nn.Conv2d(base_channels, base_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels // 2),
            nn.ReLU(inplace=True),
            ResidualBlock(base_channels // 2),
            nn.Conv2d(base_channels // 2, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, img_rgb):
        # R/G/B
        x_r = img_rgb[:, 0:1, :, :]
        x_g = img_rgb[:, 1:2, :, :]
        x_b = img_rgb[:, 2:3, :, :]

        # feature extraction
        feat_r = self.unet_R(x_r)
        feat_g = self.unet_G(x_g)
        feat_b = self.unet_B(x_b)

        # fusion
        fused = torch.cat([feat_r, feat_g, feat_b], dim=1)
        shared_feat = self.fusion_net(fused)

        # output
        height_logits = self.height_decoder(shared_feat)
        reflectance = self.reflectance_decoder(shared_feat)

        # sigmoid
        height = self.height_scale * torch.sigmoid(height_logits)

        # mask
        height = height * (TF.rgb_to_grayscale(reflectance) > 0.01).float()

        return height, reflectance

