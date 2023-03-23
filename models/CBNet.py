import torch
import torch.nn as nn
import torchvision.models as models
# from ResNet import ResNet50
import torch.nn.functional as F
from einops import rearrange
from torchsummary import summary


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=1, bias=False)


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )


def conv_nxn_bn(inp, oup, kernal_size=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernal_size, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b p n (h d) -> b p h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b p h n d -> b p n (h d)')
        return self.to_out(out)


# class TransBasicBlock(nn.Module):
#     expansion = 1
#
#     def __init__(self,inplanes,planes,stride=1,upsample=None,)
#         super(TransBasicBlock, self).__init__()
#         self.conv1 = conv3x3(inplanes,inplanes)
#         self.bn1 = nn.BatchNorm2d(inplanes)
#         self.relu = nn.ReLU(inplane=True)
#         if upsample is not None and stride != 1:
#             self.conv2 = nn.ConvTranspose2d(inplanes,planes,
#                                             kernel_size=3,stride=stride,padding=1,
#                                             output_padding=1,bias=False)
#         else:
#             self.conv2 = conv3x3(inplanes,planes,stride)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.upsample = upsample
#         self.stride = stride
#
#     def forward(self,x):
#         residual = x
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#         out = self.conv2(out)
#         out = self.bn2(out)
#
#         if self.upsample is not None:
#             residual = self.upsample(x)
#         out += residual
#         out = self.relu(out)
#         return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads, dim_head, dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class MV2Block(nn.Module):
    def __init__(self, inp, oup, stride=1, expansion=4):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expansion)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expansion == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        x = x
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileViTBlock(nn.Module):
    def __init__(self, dim, depth, channel, kernel_size, patch_size, mlp_dim, dropout=0.):
        super().__init__()
        self.ph, self.pw = patch_size

        self.conv1 = conv_nxn_bn(channel, channel, kernel_size)
        self.conv2 = conv_1x1_bn(channel, dim)

        self.transformer = Transformer(dim, depth, 1, 32, mlp_dim, dropout)

        self.conv3 = conv_1x1_bn(dim, channel)
        self.conv4 = conv_nxn_bn(2 * channel, channel, kernel_size)

    def forward(self, x):
        y = x.clone()

        # Local representations
        x = self.conv1(x)
        x = self.conv2(x)

        # Global representations
        _, _, h, w = x.shape
        x = rearrange(x, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=self.ph, pw=self.pw)
        x = self.transformer(x)
        x = rearrange(x, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', h=h // self.ph, w=w // self.pw, ph=self.ph,
                      pw=self.pw)

        # Fusion
        x = self.conv3(x)
        x = torch.cat((x, y), 1)
        x = self.conv4(x)
        return x


class PConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(PConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class CA_Block(nn.Module):
    def __init__(self, in_c, b1):
        super(CA_Block, self).__init__()

        self.conv1 = conv3x3(in_c, b1)
        self.prelu = nn.PReLU()
        self.conv2 = conv3x3(b1, b1)
        self.maxpool = nn.MaxPool2d(kernel_size=1, stride=1)

    def forward(self, x):
        x0 = x
        x1 = self.conv1(x)
        x1 = self.prelu(x1)
        x1 = self.conv2(x1)
        x1 = self.maxpool(x1)
        x2 = x0 + x1
        x3 = x0 * x2
        return x3


class FeatureRefinement(nn.Module):
    def __init__(self, in_c, c1):
        super(FeatureRefinement, self).__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(in_c, c1, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            CA_Block(c1, c1),
        )
        self.b2 = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
        )
        # self.conv1 = conv3x3(in_c, c1[0])
        # self.prelu1 = nn.PReLU()
        # self.conv2 = conv3x3(c1[0], c1[1])
        # self.prelu2 = nn.PReLU()
        # self.ca = CA_Block(c1[1], c1[2])
        # self.conv3 = conv3x3(c1[2],c1[3])
        # self.prelu3 = nn.PReLU()
        self.conv1 = nn.Conv2d(c1, c1, kernel_size=1)

    def forward(self, x):
        x1 = x
        x2 = self.b1(x)
        y1 = x1 + x2
        x3 = self.b2(x2)
        y2 = y1 * x3
        z = self.conv1(y2)
        # x2 = self.prelu1(x2)
        # x2 = self.conv2(x2)
        # x2 = self.prelu2(x2)
        # x2 = self.ca(x2)
        # x3 = self.conv3(x2)
        # x3 = self.prelu3(x3)
        # y1 = x1 + x2
        # y2 = y1 * x3
        # z = self.conv4(y2)
        return z


class GCB(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(GCB, self).__init__()
        self.relu = nn.ReLU()
        self.branch0 = nn.Sequential(
            PConv2d(in_channel, out_channel, 1),
            # nn.Conv3x3(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            # nn.Conv3x3(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch1 = nn.Sequential(
            PConv2d(in_channel, out_channel, 1),
            PConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            PConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            PConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            PConv2d(in_channel, out_channel, 1),
            PConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 2)),
            PConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(2, 0)),
            PConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )

        self.branch3 = nn.Sequential(
            PConv2d(in_channel, out_channel, 1),
            # PConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 2)),
            # PConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(2, 0)),
            # PConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch4 = nn.Sequential(
            PConv2d(in_channel, out_channel, 1),
            PConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 4)),
            PConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(4, 0)),
            PConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.conv_cat = PConv2d(5 * out_channel, out_channel, 3, padding=1)
        self.conv_res = PConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)

        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3, x4), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x


class aggregation_init(nn.Module):
    def __init__(self, channel):
        super(aggregation_init, self).__init__()
        self.prelu = nn.PReLU()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = PConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = PConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = PConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = PConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = PConv2d(2 * channel, 2 * channel, 3, padding=1)

        self.conv_concat2 = PConv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat3 = PConv2d(3 * channel, 3 * channel, 3, padding=1)
        # self.conv4 = BasicConv2d(3*channel,3*channel,3,padding=1)
        # self.conv5 = nn.Conv2d(3*channel,1,1)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) \
               * self.conv_upsample3(self.upsample(x2)) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        # x = self.conv4(x3_2)
        # x = self.conv5(x)
        return x3_2


class aggregation_final(nn.Module):

    def __init__(self, channel):
        super(aggregation_final, self).__init__()
        self.prelu = nn.PReLU()

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = PConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = PConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = PConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = PConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = PConv2d(2 * channel, 2 * channel, 3, padding=1)

        self.conv_concat2 = PConv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat3 = PConv2d(3 * channel, 3 * channel, 3, padding=1)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(x1)) \
               * self.conv_upsample3(x2) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(x2_2)), 1)
        x3_2 = self.conv_concat3(x3_2)

        return x3_2


class Refine(nn.Module):
    def __init__(self):
        super(Refine, self).__init__()
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, attention, x1, x2, x3):
        # Note that there is an error in the manuscript. In the paper, the refinement strategy is depicted as ""f'=f*S1"", it should be ""f'=f+f*S1"".
        x1 = x1 + torch.mul(x1, self.upsample2(attention))
        x2 = x2 + torch.mul(x2, self.upsample2(attention))
        x3 = x3 + torch.mul(x3, attention)

        return x1, x2, x3


class MyNet(nn.Module):
    # dims = [144, 192, 240]
    # channels = [16, 32, 64, 64, 96, 96, 128, 128, 160, 160, 640]
    def __init__(self, image_size, channel, dims, channels, num_classes, expansion=4, kernel_size=3, patch_size=(2, 2)):
        super(MyNet, self).__init__()
        ih, iw = image_size
        ph, pw = patch_size
        assert ih % ph == 0 and iw % pw == 0
        L = [2, 4, 3]
        self.depth_0 = conv3x3(3, 8)
        self.layer0 = FeatureRefinement(8, 16)
        self.mvb1 = MV2Block(16, 32, 1, expansion)
        self.layer1 = FeatureRefinement(32, 64)
        self.mobileViT1 = MobileViTBlock(dims[0], L[0], 64, kernel_size, patch_size, int(dims[0] * 2))
        self.layer2 = FeatureRefinement(64, 128)
        self.mvb2 = MV2Block(128, 256, 2, expansion)
        self.layer3 = FeatureRefinement(256, 512)
        self.mobileViT2 = MobileViTBlock(dims[1], L[1], 512, kernel_size, patch_size, int(dims[1] * 4))
        self.layer4 = FeatureRefinement(512, 1024)
        self.mvb3 = MV2Block(channels[2], channels[3], 1, expansion)
        self.layer5 = FeatureRefinement(1024, 2048)
        # Decoder 1
        self.gcb_3 = GCB(512, channel)
        self.gcb_4 = GCB(1024, channel)
        self.gcb_5 = GCB(2048, channel)
        self.agg1 = aggregation_init(channel)
        # T1
        self.conv1 = PConv2d(3 * channel, 3 * channel, 3, padding=1)
        self.conv2 = conv3x3(3 * channel, 1, 1)
        self.HA = Refine()
        # Decoder 2
        self.gcb_0 = GCB(64, channel)
        self.gcb_1 = GCB(128, channel)
        self.gcb_2 = GCB(256, channel)
        self.agg2 = aggregation_final(channel)
        # T2
        self.conv3 = PConv2d(3 * channel, 3 * channel, 3, padding=1)

        # self.depth_1 = MV2Block(64,128)
        # self.depth_2 = MobileViTBlock(128,256)
        # self.depth_3 = MV2Block(256, 512)
        # self.depth_4 = MobileViTBlock(512,1024)
        # self.depth_5 = MV2Block(1024,2048)

    def forward(self, x):
        x0 = self.depth_0(x)
        f0 = self.layer0(x0)
        x1 = self.mvb1(x0)
        f1 = self.layer1(x1)
        x2 = self.mobileViT1(x1)
        f2 = self.layer2(x2)
        x3 = self.mvb2(x2)
        f3 = self.layer3(x3)
        x4 = self.mobileViT2(x3)
        f4 = self.layer4(x4)
        x5 = self.mvb3(x4)
        f5 = self.layer5(x5)

        x3 = self.gcb_3(f3)
        x4 = self.gcb_4(f4)
        x5 = self.gcb_5(f5)
        attention_map = self.agg1(x3, x4, x5)

        y1 = self.conv1(attention_map)
        y1 = self.conv2(y1)

        x0, x1, x2 = self.HA(attention_map.sigmoid(), f0, f1, f2)
        x0 = self.gcb_0(x0)
        x1 = self.gcb_1(x1)
        x2 = self.gcb_2(x2)
        y2 = self.agg2(x0, x1, x2)

        y2 = torch.cat(y2)
        y2 = self.conv3(y2)

        y = y1 + y2
        return y


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def count():
    dims = [144, 192, 240]
    channels = [16, 32, 64, 64, 96, 96, 128, 128, 160, 160, 640]
    return MyNet((224, 224), 64, dims, channels, num_classes=1000)


if __name__ == '__main__':
    img = torch.randn(1, 3, 224, 224)
    model = count()
    # out = model(img)
    # print(out.shape)
    # model = MyNet((224, 224),32, dims, channels, num_classes=1000).to(device)
    # t = torch.rand(3,224,224)
    # model(t)
    summary(model, (3, 224, 224)),
