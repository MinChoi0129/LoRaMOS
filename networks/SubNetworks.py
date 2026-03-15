import torch
import torch.nn as nn
import torch.nn.functional as F
from networks import backbone_moving
from networks.backbone_movable import MetaKernel, ResContextBlock, ResBlock, UpBlock


class MovingNet(nn.Module):
    """Cartesian BEV Encoder-Decoder."""

    def __init__(self, in_channels):
        super(MovingNet, self).__init__()
        block = backbone_moving.BasicBlock

        # ---- Encoder ----
        self.enc1 = self._make_layer(block, in_channels, 64, num_blocks=3, stride=2)  # → [B, 64, 256, 256]
        self.enc2 = self._make_layer(block, 64, 128, num_blocks=3, stride=2)  # → [B, 128, 128, 128]
        self.enc3 = self._make_layer(block, 128, 256, num_blocks=4, stride=2)  # → [B, 256, 64, 64]

        # ---- Deformable Attention at bottleneck ----
        self.bottleneck_attn = backbone_moving.DeformAttnBottleneck(
            in_channels=256,
            d_model=128,
            d_ffn=512,
            n_heads=4,
            n_points=4,
            num_layers=2,
        )

        # ---- Decoder (Skip Connection) ----
        self.dec3 = backbone_moving.BasicConv2d(256 + 128, 128, kernel_size=3, padding=1)  # attn + enc2 skip
        self.dec2 = backbone_moving.BasicConv2d(128 + 64, 64, kernel_size=3, padding=1)    # dec3 + enc1 skip
        self.dec1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=64, bias=False),
            nn.Conv2d(64, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

    def _make_layer(self, block, in_planes, out_planes, num_blocks, stride=2, dilation=1):
        layer = []
        layer.append(backbone_moving.DownSample2D(in_planes, out_planes, stride=stride))
        for i in range(num_blocks):
            layer.append(block(out_planes, dilation=dilation, use_att=False))
        layer.append(block(out_planes, dilation=dilation, use_att=True))
        return nn.Sequential(*layer)

    def forward(self, bev_feat, movable_probability_mask_bev):
        """
        bev_feat:                       [B, T*64, 512, 512]  누적 BEV feature
        movable_probability_mask_bev:   [B, 1, 512, 512]     객체 존재 확률
        """

        # ---- Encoder ----
        e1 = self.enc1(bev_feat)  # [B, 64, 256, 256]
        m1 = F.max_pool2d(movable_probability_mask_bev, kernel_size=2, stride=2)
        e1 = e1 * (1 + m1)

        e2 = self.enc2(e1)  # [B, 128, 128, 128]
        m2 = F.max_pool2d(m1, kernel_size=2, stride=2)
        e2 = e2 * (1 + m2)

        e3 = self.enc3(e2)  # [B, 256, 64, 64]

        # ---- Deformable Attention ----
        e3 = self.bottleneck_attn(e3)  # [B, 256, 64, 64]

        # ---- Decoder: Skip Connection ----
        d3 = F.interpolate(e3, size=e2.shape[2:], mode="bilinear", align_corners=True)
        d3 = self.dec3(torch.cat([d3, e2], dim=1))  # [B, 128, 128, 128]

        d2 = F.interpolate(d3, size=e1.shape[2:], mode="bilinear", align_corners=True)
        d2 = self.dec2(torch.cat([d2, e1], dim=1))  # [B, 64, 256, 256]

        d1 = self.dec1(d2)  # [B, 32, 512, 512]

        return d1


class MovableNet(nn.Module):
    def __init__(self, in_ch=5, out_ch=3, num_batch=6, height=64, width=2048):
        super(MovableNet, self).__init__()

        # Context Blocks: range image → 32ch feature
        self.downCntx = ResContextBlock(in_ch, 32)
        self.metaConv = MetaKernel(num_batch=num_batch, feat_height=height, feat_width=width, coord_channels=in_ch)
        self.downCntx2 = ResContextBlock(32, 32)
        self.downCntx3 = ResContextBlock(32, 32)

        # Encoder: 4 stages with (2,4) pooling
        self.resBlock1 = ResBlock(32, 2 * 32, 0.2, pooling=True, drop_out=False, kernel_size=(2, 4))
        self.resBlock2 = ResBlock(2 * 32, 2 * 2 * 32, 0.2, pooling=True, kernel_size=(2, 4))
        self.resBlock3 = ResBlock(2 * 2 * 32, 2 * 4 * 32, 0.2, pooling=True, kernel_size=(2, 4))
        self.resBlock4 = ResBlock(2 * 4 * 32, 2 * 4 * 32, 0.2, pooling=True, kernel_size=(2, 4))

        # Decoder: 3 UpBlocks with PixelShuffle(2,4) + skip connections
        self.upBlock1 = UpBlock(2 * 4 * 32, 4 * 32, 0.2)
        self.upBlock2 = UpBlock(4 * 32, 2 * 32, 0.2)
        self.upBlock3 = UpBlock(2 * 32, 32, 0.2, drop_out=False)

        # Movable logits
        self.logits = nn.Conv2d(32, out_ch, kernel_size=(1, 1))

    def forward(self, rv_input):
        # rv_input: [B, 5, 64, 2048]
        B = rv_input.size(0)
        self.metaConv.update_num_batch(B)

        # 1. Context with MetaKernel geometry-awareness
        x = self.downCntx(rv_input)  # [B, 32, 64, 2048]
        x = self.metaConv(
            data=x, coord_data=rv_input, data_channels=x.size(1), coord_channels=rv_input.size(1), kernel_size=3
        )  # [B, 32, 64, 2048]
        x = self.downCntx2(x)  # [B, 32, 64, 2048]
        x = self.downCntx3(x)  # [B, 32, 64, 2048]

        # 2. Encoder
        down0c, down0b = self.resBlock1(x)  # [B, 64, 32, 512],  skip: [B, 64, 64, 2048]
        down1c, down1b = self.resBlock2(down0c)  # [B, 128, 16, 128], skip: [B, 128, 32, 512]
        down2c, down2b = self.resBlock3(down1c)  # [B, 256, 8, 32],   skip: [B, 256, 16, 128]
        down3c, down3b = self.resBlock4(down2c)  # [B, 256, 4, 8],    skip: [B, 256, 8, 32]

        # 3. Decoder with skip connections
        up3 = self.upBlock1(down3b, down2b)  # [B, 128, 16, 128]
        up2 = self.upBlock2(up3, down1b)  # [B, 64, 32, 512]
        up1 = self.upBlock3(up2, down0b)  # [B, 32, 64, 2048]

        # 4. Movable logits (raw — softmax는 loss에서 적용)
        logits = self.logits(up1)  # [B, out_ch, 64, 2048]
        return logits
