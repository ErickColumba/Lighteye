"""BiSeNet para análisis facial (face parsing): qué píxel es piel, pelo, ojos…

Arquitectura de facexlib (MIT), compatible con `parsing_bisenet.pth`. Trabaja
sobre la cara alineada a 512×512 (la misma que se usa para restaurar).
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

# Clases que devuelve el modelo.
CLASSES = ["fondo", "piel", "ceja_izq", "ceja_der", "ojo_izq", "ojo_der", "gafas", "oreja_izq",
           "oreja_der", "pendiente", "nariz", "boca", "labio_sup", "labio_inf", "cuello",
           "collar", "ropa", "pelo", "sombrero"]
SKIN = (1, 10, 14)  # piel de la cara, nariz y cuello
EYES = (4, 5)
BROWS = (2, 3)
LIPS = (12, 13)
HAIR = (17,)


class ConvBNReLU(nn.Module):
    def __init__(self, cin, cout, ks=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, ks, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(cout)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.downsample = None
        if cin != cout or stride != 1:
            self.downsample = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False),
                                            nn.BatchNorm2d(cout))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        shortcut = x if self.downsample is None else self.downsample(x)
        return F.relu(out + shortcut)


class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.layer1 = nn.Sequential(BasicBlock(64, 64), BasicBlock(64, 64))
        self.layer2 = nn.Sequential(BasicBlock(64, 128, 2), BasicBlock(128, 128))
        self.layer3 = nn.Sequential(BasicBlock(128, 256, 2), BasicBlock(256, 256))
        self.layer4 = nn.Sequential(BasicBlock(256, 512, 2), BasicBlock(512, 512))

    def forward(self, x):
        x = self.maxpool(F.relu(self.bn1(self.conv1(x))))
        feat8 = self.layer2(self.layer1(x))
        feat16 = self.layer3(feat8)
        feat32 = self.layer4(feat16)
        return feat8, feat16, feat32


class AttentionRefinement(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv = ConvBNReLU(cin, cout)
        self.conv_atten = nn.Conv2d(cout, cout, 1, bias=False)
        self.bn_atten = nn.BatchNorm2d(cout)

    def forward(self, x):
        feat = self.conv(x)
        atten = torch.sigmoid(self.bn_atten(self.conv_atten(F.adaptive_avg_pool2d(feat, 1))))
        return feat * atten


class ContextPath(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = ResNet18()
        self.arm16 = AttentionRefinement(256, 128)
        self.arm32 = AttentionRefinement(512, 128)
        self.conv_head32 = ConvBNReLU(128, 128)
        self.conv_head16 = ConvBNReLU(128, 128)
        self.conv_avg = ConvBNReLU(512, 128, 1, 1, 0)

    def forward(self, x):
        feat8, feat16, feat32 = self.resnet(x)
        avg = self.conv_avg(F.adaptive_avg_pool2d(feat32, 1))
        feat32_sum = self.arm32(feat32) + F.interpolate(avg, feat32.shape[2:], mode="nearest")
        feat32_up = self.conv_head32(F.interpolate(feat32_sum, feat16.shape[2:], mode="nearest"))
        feat16_sum = self.arm16(feat16) + feat32_up
        feat16_up = self.conv_head16(F.interpolate(feat16_sum, feat8.shape[2:], mode="nearest"))
        return feat8, feat16_up, feat32_up


class FeatureFusion(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.convblk = ConvBNReLU(cin, cout, 1, 1, 0)
        self.conv1 = nn.Conv2d(cout, cout // 4, 1, bias=False)
        self.conv2 = nn.Conv2d(cout // 4, cout, 1, bias=False)

    def forward(self, fsp, fcp):
        feat = self.convblk(torch.cat([fsp, fcp], dim=1))
        atten = torch.sigmoid(self.conv2(F.relu(self.conv1(F.adaptive_avg_pool2d(feat, 1)))))
        return feat * atten + feat


class Output(nn.Module):
    def __init__(self, cin, mid, classes):
        super().__init__()
        self.conv = ConvBNReLU(cin, mid)
        self.conv_out = nn.Conv2d(mid, classes, 1, bias=False)

    def forward(self, x):
        return self.conv_out(self.conv(x))


class BiSeNet(nn.Module):
    def __init__(self, classes: int = 19):
        super().__init__()
        self.cp = ContextPath()
        self.ffm = FeatureFusion(256, 256)
        self.conv_out = Output(256, 256, classes)
        self.conv_out16 = Output(128, 64, classes)  # solo para entrenar; se carga igual
        self.conv_out32 = Output(128, 64, classes)

    def forward(self, x):
        feat8, feat_cp8, _ = self.cp(x)
        out = self.conv_out(self.ffm(feat8, feat_cp8))
        return F.interpolate(out, x.shape[2:], mode="bilinear", align_corners=True)


_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)


def load(path: str, device: str) -> BiSeNet:
    net = BiSeNet()
    net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    return net.to(device).eval()


def parse(net: BiSeNet, face: np.ndarray) -> np.ndarray:
    """Etiqueta (512×512 uint8) de cada píxel de una cara alineada (sRGB 0–1)."""
    x = (face.astype(np.float32) - _MEAN) / _STD
    p = next(net.parameters())
    t = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))[None].to(p.device, p.dtype)
    with torch.inference_mode():
        return net(t)[0].argmax(0).byte().cpu().numpy()
