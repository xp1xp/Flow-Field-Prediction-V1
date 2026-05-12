# Unet 改进版本V1 == MedNeXt_UNet_Lift
# 使用前改名为 unet_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_num_groups(num_channels: int, max_groups: int = 8) -> int:
    """
    Return a valid GroupNorm group number that divides num_channels.
    """
    for g in range(min(max_groups, num_channels), 0, -1):
        if num_channels % g == 0:
            return g
    return 1


class GRN(nn.Module):
    """
    Global Response Normalization
    Reference idea from ConvNeXt V2, adapted for 2D feature maps.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.eps = eps

    def forward(self, x):
        # x: [B, C, H, W]
        gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)                 # [B, C, 1, 1]
        nx = gx / (gx.mean(dim=1, keepdim=True) + self.eps)               # [B, C, 1, 1]
        return x + self.gamma * (x * nx) + self.beta


class MedNeXtBlock2D(nn.Module):
    """
    MedNeXt-style block:
    depthwise large-kernel conv -> GN -> 1x1 expand -> GELU -> GRN -> 1x1 project -> residual
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 7,
        expansion_ratio: int = 4,
        max_groups: int = 8,
        use_grn: bool = True,
    ):
        super().__init__()

        hidden_channels = in_channels * expansion_ratio
        padding = kernel_size // 2
        groups = _get_num_groups(in_channels, max_groups)

        self.dwconv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
            bias=False,
        )
        self.norm = nn.GroupNorm(groups, in_channels)
        self.pw_expand = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=True)
        self.act = nn.GELU()
        self.grn = GRN(hidden_channels) if use_grn else nn.Identity()
        self.pw_project = nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=True)

        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        identity = self.shortcut(x)

        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pw_expand(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pw_project(x)

        return x + identity


class DoubleConv(nn.Module):
    """
    Keep the old class name for compatibility,
    but replace the internal implementation with MedNeXt-style blocks.
    """
    def __init__(self, in_channels, out_channels, kernel_size=7):
        super().__init__()
        self.block = nn.Sequential(
            MedNeXtBlock2D(in_channels, out_channels, kernel_size=kernel_size),
            MedNeXtBlock2D(out_channels, out_channels, kernel_size=kernel_size),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    """
    Residual learned downsampling instead of MaxPool + DoubleConv.
    Keep the class name unchanged for easy replacement.
    """
    def __init__(self, in_channels, out_channels, kernel_size=5):
        super().__init__()

        self.downsample_main = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.downsample_skip = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=2, bias=False
        )

        self.block1 = MedNeXtBlock2D(out_channels, out_channels, kernel_size=kernel_size)
        self.block2 = MedNeXtBlock2D(out_channels, out_channels, kernel_size=kernel_size)

    def forward(self, x):
        x = self.downsample_main(x) + self.downsample_skip(x)
        x = self.block1(x)
        x = self.block2(x)
        return x


class Up(nn.Module):
    """
    Bilinear upsample + skip fusion + MedNeXt blocks.
    Keep the class name unchanged for compatibility.
    """
    def __init__(self, in_channels, out_channels, bilinear=True, skip_channels=None, kernel_size=5):
        super().__init__()
        self.bilinear = bilinear

        if skip_channels is None:
            raise ValueError("skip_channels must be provided in this MedNeXt Up block.")

        self.reduce = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.fuse = nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=1, bias=False)

        self.block1 = MedNeXtBlock2D(out_channels, out_channels, kernel_size=kernel_size)
        self.block2 = MedNeXtBlock2D(out_channels, out_channels, kernel_size=kernel_size)

    def forward(self, x1, x2):
        # x1: decoder feature, x2: skip feature
        x1 = F.interpolate(x1, size=x2.shape[-2:], mode='bilinear', align_corners=False)
        x1 = self.reduce(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.fuse(x)
        x = self.block1(x)
        x = self.block2(x)
        return x


class HeightLiftingHead(nn.Module):
    """
    Learned lifting decoder:
    [B, C, 48, 48] -> [B, C, 64, 48] -> [B, 3, 64, 48]

    It replaces the original fixed bilinear resize at the output.
    """
    def __init__(self, in_channels=64, hidden_channels=64, out_channels=3, in_height=48, out_height=64):
        super().__init__()
        self.in_height = in_height
        self.out_height = out_height

        gn1 = _get_num_groups(hidden_channels, 8)
        gn2 = _get_num_groups(hidden_channels, 8)

        self.pre = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(gn1, hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(gn2, hidden_channels),
            nn.GELU(),
        )

        # shared height-direction mapping: 48 -> 64
        self.height_proj = nn.Sequential(
            nn.Linear(in_height, 96),
            nn.GELU(),
            nn.Linear(96, out_height),
        )

        self.post = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_get_num_groups(hidden_channels, 8), hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=True),
        )

    def forward(self, x):
        # x: [B, C, 48, 48]
        x = self.pre(x)

        # If training/inference input size changes unexpectedly,
        # fallback to interpolation before learned lifting.
        if x.shape[2] != self.in_height:
            x = F.interpolate(x, size=(self.in_height, x.shape[3]), mode='bilinear', align_corners=False)

        # Apply shared MLP along height dimension
        # [B, C, H, W] -> [B, C, W, H]
        x = x.permute(0, 1, 3, 2).contiguous()
        x = self.height_proj(x)  # [B, C, W, 64]
        x = x.permute(0, 1, 3, 2).contiguous()  # [B, C, 64, W]

        x = self.post(x)  # [B, 3, 64, 48]
        return x


class UNet(nn.Module):
    """
    MedNeXt-UNet-Lift version
    Input : [B, 2, 48, 48]
    Output: [B, 3, 64, 48]

    External interface is intentionally kept compatible with the original code.
    """
    def __init__(self, n_channels=2, n_classes=3, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear  # kept for compatibility

        # stem / encoder
        self.inc = nn.Sequential(
            nn.Conv2d(n_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_get_num_groups(64, 8), 64),
            nn.GELU(),
            MedNeXtBlock2D(64, 64, kernel_size=7),
            MedNeXtBlock2D(64, 64, kernel_size=7),
        )

        self.down1 = Down(64, 128, kernel_size=7)   # 48 -> 24
        self.down2 = Down(128, 256, kernel_size=5)  # 24 -> 12
        self.down3 = Down(256, 512, kernel_size=3)  # 12 -> 6
        self.down4 = Down(512, 512, kernel_size=3)  # 6 -> 3

        # bottleneck refinement
        self.bottleneck = nn.Sequential(
            MedNeXtBlock2D(512, 512, kernel_size=3),
            MedNeXtBlock2D(512, 512, kernel_size=3),
            MedNeXtBlock2D(512, 512, kernel_size=3),
        )

        # decoder
        self.up1 = Up(512, 256, bilinear=bilinear, skip_channels=512, kernel_size=3)  # 3 -> 6
        self.up2 = Up(256, 128, bilinear=bilinear, skip_channels=256, kernel_size=5)  # 6 -> 12
        self.up3 = Up(128, 64, bilinear=bilinear, skip_channels=128, kernel_size=7)   # 12 -> 24
        self.up4 = Up(64, 64, bilinear=bilinear, skip_channels=64, kernel_size=7)      # 24 -> 48

        # learned lifting output head: 48 -> 64
        self.lift_head = HeightLiftingHead(
            in_channels=64,
            hidden_channels=64,
            out_channels=n_classes,
            in_height=48,
            out_height=64
        )

    def forward(self, x):
        # encoder
        x1 = self.inc(x)       # [B, 64, 48, 48]
        x2 = self.down1(x1)    # [B, 128, 24, 24]
        x3 = self.down2(x2)    # [B, 256, 12, 12]
        x4 = self.down3(x3)    # [B, 512, 6, 6]
        x5 = self.down4(x4)    # [B, 512, 3, 3]

        x5 = self.bottleneck(x5)

        # decoder
        x = self.up1(x5, x4)   # [B, 256, 6, 6]
        x = self.up2(x, x3)    # [B, 128, 12, 12]
        x = self.up3(x, x2)    # [B, 64, 24, 24]
        x = self.up4(x, x1)    # [B, 64, 48, 48]

        # learned lifting output
        logits = self.lift_head(x)   # [B, 3, 64, 48]
        return logits


def get_unet_model(n_channels=2, n_classes=3):
    return UNet(n_channels=n_channels, n_classes=n_classes)


if __name__ == "__main__":
    # quick shape check
    model = get_unet_model(n_channels=2, n_classes=3)
    x = torch.randn(2, 2, 48, 48)
    y = model(x)
    print("input :", x.shape)
    print("output:", y.shape)  # expected: [2, 3, 64, 48]