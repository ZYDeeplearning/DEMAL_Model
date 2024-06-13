import cv2
import torch
from torch import nn
from torch.nn import Parameter
from torch.nn import functional as F
from torch.nn.utils import spectral_norm
from utils import *
from torchvision import transforms
# from torchvision.transforms import InterpolationMode
# from torch.utils import data


# channle_wise
class layer_norm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.layer_norm(x, x.size()[1:])


class Conv2D(nn.Module):
    def __init__(self, in_channels=256, out_channels=256, kernel_size=3, strides=1, padding=1, sn=False):
        super().__init__()
        self.sn = sn
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=strides, padding=padding,
                              padding_mode="reflect")

    def forward(self, x):
        return self.conv(x)


# 定义卷积
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, strides=1, padding=0, sn=False, bias=False,
                 padding_mode='reflect'):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=strides, padding=padding, bias=bias,
                              padding_mode=padding_mode)
        self.ins_norm = layer_norm()
        self.activation = nn.LeakyReLU(0.2, True)

        init_weights(self)

    def forward(self, x):
        out = self.conv(x)
        out = self.ins_norm(out)
        out = self.activation(out)

        return out


class Discriminator_T(nn.Module):
    def __init__(self):
        super(Discriminator_T, self).__init__()
        self.channels = [1, 32, 64, 128, 256]
        self.model = []
        self.model += [ConvBlock(in_channels=self.channels[0], out_channels=self.channels[1], padding=1, sn=True),
                       ConvBlock(in_channels=self.channels[1], out_channels=self.channels[2], strides=2, padding=1,
                                 sn=True),
                       ConvBlock(in_channels=self.channels[2], out_channels=self.channels[3], strides=2, padding=1,
                                 sn=True),
                       ConvBlock(in_channels=self.channels[3], out_channels=self.channels[4], strides=2, padding=1,
                                 sn=True),
                       Conv2D(in_channels=self.channels[4], out_channels=1, padding=1, sn=True)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        # 3*1
        out = self.model(x)
        return out


# 定义生成器
class ILN(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(ILN, self).__init__()
        self.eps = eps
        self.rho = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.gamma = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.beta = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.rho.data.fill_(0.0)
        self.gamma.data.fill_(1.0)
        self.beta.data.fill_(0.0)

    def forward(self, input):
        in_mean, in_var = torch.mean(input, dim=[2, 3], keepdim=True), torch.var(input, dim=[2, 3], keepdim=True)
        out_in = (input - in_mean) / torch.sqrt(in_var + self.eps)
        ln_mean, ln_var = torch.mean(input, dim=[1, 2, 3], keepdim=True), torch.var(input, dim=[1, 2, 3], keepdim=True)
        out_ln = (input - ln_mean) / torch.sqrt(ln_var + self.eps)
        out = self.rho.expand(input.shape[0], -1, -1, -1) * out_in + (
                    1 - self.rho.expand(input.shape[0], -1, -1, -1)) * out_ln
        out = out * self.gamma.expand(input.shape[0], -1, -1, -1) + self.beta.expand(input.shape[0], -1, -1, -1)
        return out

def gram_matrix(y):
	(b, ch, h, w) = y.size()
	features = y.reshape(b, ch, w * h)
	features_t = features.transpose(1, 2)
	gram = features.bmm(features_t) / (ch * h * w)
	return gram

def mean_variance_norm(feat):
    size = feat.size()
    mean, std = calc_mean_std(feat)
    normalized_feat = (feat - mean.expand(size)) / std.expand(size)
    return normalized_feat

# 残差的SM
# class utm(nn.Module):
#     def __init__(self):
#         super(utm, self).__init__()
#
#         self.net = nn.Sequential(nn.Conv2d(256,128,1,1,0),
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(128,32,1,1,0))
#         self.uncompress = nn.Conv2d(32,256,1,1,0)
#         self.sm = nn.Softmax(dim=-1)
#     def forward(self, content, style=None,noise=None,init=False):
#         if init:
#             cF_nor = nor_mean_std(content)
#             cF = self.net(cF_nor)
#             cF = self.uncompress(cF)
#             cF = cF +content
#             return cF
#         else:
#             cF_nor = nor_mean_std(content)
#             sF_nor, smean = nor_mean(style)
#             cF = self.net(cF_nor)
#             sF = self.net(sF_nor)
#             b, c, w, h = cF.size()
#             s_cov = gram_matrix(sF)
#             s_cov = self.sm(s_cov)
#             gF = torch.bmm(s_cov, cF.flatten(2, 3)).view(b,c,w,h)
#             gF = self.uncompress(gF)
#             if noise==None:
#                 gF = gF + smean.expand(cF_nor.size())+content+style
#             else:
#                 gF = gF + smean.expand(cF_nor.size())+content+noise
#             return gF
# 改进残差的SM
class utm(nn.Module):
    def __init__(self):
        super(utm, self).__init__()

        self.net = nn.Sequential(nn.Conv2d(256,128,1,1,0),
                nn.ReLU(inplace=True),
                nn.Conv2d(128,32,1,1,0))
        self.uncompress = nn.Conv2d(32,256,1,1,0)
        self.sm = nn.Softmax(dim=-1)
        self.adain = AdaIN()
    def forward(self, content, style=None,noise=None,init=False):
        if init:
            cF_nor = nor_mean_std(content)
            cF = self.net(cF_nor)
            cF = self.uncompress(cF)
            cF = cF +content
            return cF
        else:

            cF_nor = nor_mean_std(content)
            sF_nor, smean = nor_mean(style)
            cF = self.net(cF_nor)
            sF = self.net(sF_nor)
            b, c, w, h = cF.size()
            # gF =self.adain(cF, sF)
            s_cov = calc_cov(sF)
            b1, c1, hw = s_cov.size()
            s_cov += self.sm(s_cov) * int(c1) ** (-0.5)  # test
            gF = torch.bmm(s_cov,cF.flatten(2, 3)).view(b,c,w,h)
            gF = self.uncompress(gF)
            if noise==None:
                gF = gF + smean.expand(cF_nor.size())
                gF = 1 * gF + 1 * content
            else:
                gF = gF + smean.expand(cF_nor.size())+content
            return gF

class AdaIN(nn.Module):
    def __init__(self, epsilon=1e-5):
        super(AdaIN, self).__init__()
        self.epsilon = epsilon

    def forward(self, x, y):
        content_mean, content_std = calc_mean_std(x)
        y_beta, y_gamma = calc_mean_std(y)  # sty_std *(c_features - c_mean) / c_std + sty_mean
        normalized_features = y_gamma * (x - content_mean) / content_std + y_beta
        return normalized_features

class attention(nn.Module):
    def __init__(self, in_planes=256, max_sample=256 * 256):
        super(attention, self).__init__()
        self.q = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.k = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.v = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.sm = nn.Softmax(dim=-1)
        self.out_conv1 = nn.Conv2d(in_planes, in_planes, (1, 1))
    def forward(self, content, style):
        c = mean_variance_norm(content)
        s = mean_variance_norm(style)
        AG_q = self.q(c)
        s_k = self.k(s)
        s_v = self.v(style)

        b, c, h, w = AG_q.size()
        AG_q = AG_q.view(b, -1, w * h)  # C x HsWs
        b, c, h, w = s_k.size()
        s_k = s_k.view(b, -1, w * h).permute(0, 2, 1)  # HsWs x C
        AS = torch.bmm(AG_q, s_k)  # C x C
        AS = self.sm(AS)  # aesthetic attention map

        b, c, h, w = s_v.size()
        s_v = s_v.view(b, -1, w * h)  # C x HsWs
        astyle = torch.bmm(AS, s_v)  # C x HsWs

        DA = astyle.view(b, c, h, w)
        DA = self.out_conv1(DA)
        return DA+content

class GuideDiscriminator(nn.Module):
    def __init__(self, in_channels):
        super(GuideDiscriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalize=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=False))
            return layers

        # Construct three discriminator models
        self.models = nn.ModuleList()
        self.score_models = nn.ModuleList()
        for i in range(3):
            self.models.append(
                nn.Sequential(
                    *discriminator_block(in_channels, 64, normalize=False),
                    *discriminator_block(64, 128),
                    *discriminator_block(128, 256),
                    *discriminator_block(256, 512)
                )
            )
            self.score_models.append(
                nn.Sequential(
                    nn.Conv2d(512, 1, 3, padding=1)
                )
            )

        self.downsample = nn.AvgPool2d(in_channels, stride=2, padding=[1, 1], count_include_pad=False)

    # Compute the MSE between model output and scalar gt
    def compute_loss(self, x, gt):
        _, outputs = self.forward(x)

        loss = sum([torch.mean((out - gt) ** 2) for out in outputs])
        return loss

    def forward(self,  x, x_low = None):
        outputs = []
        feats = []
        feats_low = []
        for i in range(len(self.models)):
            feats.append(self.models[i](x))
            outputs.append(self.score_models[i](self.models[i](x)))
            x = self.downsample(x)
        if x_low != None:
            for j in range(len(self.models) -1):
                feats_low.append(self.models[j](x_low)) #`16*16, 8*8
                x_low = self.downsample(x_low)

        self.upsample = nn.Upsample(size=(feats[0].size()[2], feats[0].size()[3]), mode='nearest')
        feat = feats[0]
        if x_low != None:
            for i in range(1, len(feats)):
                feat = feat + self.upsample(feats_low[i-1]) + self.upsample(feats[i])
        else:
            for i in range(1, len(feats)):
                feat = feat +  self.upsample(feats[i])
        return feat, outputs
# 定义残差块
class ResBlock(nn.Module):
    def __init__(self, channels, use_bias=False):
        super().__init__()
        Res_block = []
        Res_block += [nn.ReflectionPad2d(1),
                      nn.Conv2d(channels, channels, 3, 1, 0, bias=use_bias),
                      ILN(channels), nn.PReLU(num_parameters=1)]

        Res_block += [nn.ReflectionPad2d(1),
                      nn.Conv2d(channels, channels, 3, 1, 0, bias=use_bias),
                      ILN(channels)]
        self.Res_block = nn.Sequential(*Res_block)

    def forward(self, x):
        return x + self.Res_block(x)


class StyleEncoder(nn.Module):
    def __init__(self, img_channels=3, num_features=64, padding_mode="reflect", ):
        super().__init__()
        self.padding_mode = padding_mode

        self.initial_down = nn.Sequential(
            # k7n32s1
            nn.Conv2d(img_channels, num_features, kernel_size=7, stride=1, padding=3,
                      padding_mode=self.padding_mode),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        # Down-convolution
        self.down1 = nn.Sequential(
            # k3n32s2
            nn.Conv2d(num_features, num_features, kernel_size=3, stride=2, padding=1,
                      padding_mode=self.padding_mode),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # k3n64s1
            nn.Conv2d(num_features, num_features * 2, kernel_size=3, stride=1, padding=1,
                      padding_mode=self.padding_mode),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        self.down2 = nn.Sequential(
            # k3n64s2
            nn.Conv2d(num_features * 2, num_features * 2, kernel_size=3, stride=2, padding=1,
                      padding_mode=self.padding_mode),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # k3n128s1
            nn.Conv2d(num_features * 2, num_features * 4, kernel_size=3, stride=1, padding=1,
                      padding_mode=self.padding_mode),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

    def forward(self, x):
        style_code = self.down2(self.down1(self.initial_down(x)))

        return style_code


# 定义生成器-Encode
class GenEncoder(nn.Module):
    def __init__(self, hw=64, n_block=5, norm=nn.InstanceNorm2d, use_bias=False):
        super().__init__()
        # 平面卷积
        model = []
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(3, hw, 7, 1, 0, bias=use_bias)]
        # 下采样
        down = 2
        for i in range(down):
            mult = 2 ** i
            model += [nn.Conv2d(hw * mult, hw * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      ILN(hw * mult * 2), nn.ReLU(True)]
            # 残差块
        res = hw * 4
        for j in range(n_block):
            model += [ResBlock(res)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)  # 1,256,64,64


# 定义解码器
class GenDecoder(nn.Module):
    def __init__(self, hw=64, out_channels=3, n_block=4, use_bias=False):
        super().__init__()
        # 残差块
        model = []
        res = hw * 4
        for i in range(n_block):
            model += [ResBlock(res)]
        # frist upsample
        mult = 2 ** (n_block // 2)
        model += [nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                  nn.Conv2d(int(hw * mult), int(hw * mult / 2), kernel_size=3, stride=1, padding=1),
                  nn.Conv2d(int(hw * mult / 2), int(hw * mult / 2), kernel_size=3, stride=1, padding=1),
                  ILN(int(hw * mult / 2)),
                  nn.ReLU(True)]
        # second upsampling
        model += [nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                  nn.Conv2d(int(hw * mult / 2), int(hw * mult / 4), kernel_size=3, stride=1, padding=1),
                  nn.Conv2d(int(hw * mult / 4), int(hw * mult / 4), kernel_size=3, stride=1, padding=1),
                  ILN(int(hw * mult / 4)),
                  nn.ReLU(True)]
        # addtional layer
        model += [nn.Conv2d(int(hw * mult / 4), int(hw * mult / 8), kernel_size=3, stride=1, padding=1),
                  nn.Conv2d(int(hw * mult / 8), int(hw * mult / 8), kernel_size=3, stride=1, padding=1),
                  ILN(int(hw * mult / 8)), nn.ReLU(True)]
        model += [nn.Conv2d(int(hw * mult / 8), int(hw * mult / 16), kernel_size=3, stride=1, padding=1),
                  nn.Conv2d(int(hw * mult / 16), int(hw * mult / 16), kernel_size=3, stride=1, padding=1),
                  ILN(int(hw * mult / 16)), nn.ReLU(True)]
        model += [nn.Conv2d(int(hw * mult / 16), out_channels, 7, 1, 3), nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)
    # 定义生成器


# 定义生成器
# 定义生成器
class Generator(nn.Module):
    def __init__(self, n_domian=None, E_block=5, D_block=4):
        super(Generator, self).__init__()
        # 编码器
        self.Encoder = [GenEncoder(n_block=E_block)]
        self.Encoder = nn.Sequential(*self.Encoder)
        # 解码器
        self.Decoder = [GenDecoder(n_block=D_block)]
        self.Decoder = nn.Sequential(*self.Decoder)

    def encoder(self, x):
        return self.Encoder(x)  # type: ignore

    def decoders(self, x):
        return self.Decoder(x)

    def forward(self, x):
        encode = self.encoder(x)
        return self.decoders(encode)


# img _level D
class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, act=True):
        super().__init__()
        self.act = act
        self.sn_conv = spectral_norm(nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            padding_mode="zeros"
            # Author's code used slim.convolution2d, which is using SAME padding (zero padding in pytorch)
        ))
        self.LReLU = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x = self.sn_conv(x)
        if self.act:
            x = self.LReLU(x)
        return x


class Discriminator_S(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=[32, 64, 128]):
        super().__init__()

        self.model = nn.Sequential(

            # k3n32s2
            Block(in_channels, features[0], kernel_size=3, stride=2, padding=1),  # b,32,128,128
            # k3n32s1
            Block(features[0], features[0], kernel_size=3, stride=1, padding=1),  # b,32,128,128

            # k3n64s2
            Block(features[0], features[1], kernel_size=3, stride=2, padding=1),  # b,64,64,64
            # k3n64s1
            Block(features[1], features[1], kernel_size=3, stride=1, padding=1),  # b,64,64,64

            # k3n128s2
            Block(features[1], features[2], kernel_size=3, stride=2, padding=1),  # b,128,32,32
            # k3n128s1
            Block(features[2], features[2], kernel_size=3, stride=1, padding=1),  # b,128,32,32

            # k1n1s1
            Block(features[2], out_channels, kernel_size=1, stride=1, padding=0, act=False)  # b,3,32,32
        )

    def forward(self, x, y=0):
        x = self.model(x)
        return x


# #patchgan
# class PatchGAN_D(nn.Module):
#           def __init__(self,hw=64,out_channels=1,norm=nn.InstanceNorm2d,use_bias=False,use_sn=True):
#                   super(PatchGAN_D,self).__init__()
#                   model=[]
#                   self.use_sn=use_sn
#                   #平面卷积
#                   model+=[nn.Conv2d(out_channels,hw,kernel_size=3,stride=1,padding=1,bias=True),
#                           nn.LeakyReLU(0.2,True)
#                   ]
#                   #下采样
#                   model+=[nn.Conv2d(hw,hw*2,kernel_size=3,stride=2,padding=1,bias=True),
#                               nn.LeakyReLU(0.2,True),nn.Conv2d(hw*2,hw*4,kernel_size=3,stride=1,padding=1,bias=True),
#                               norm(hw*4),nn.LeakyReLU(0.2,True),
#                               nn.Conv2d(hw*4,hw*4,kernel_size=3,stride=2,padding=1,bias=True),
#                               nn.LeakyReLU(0.2,True),nn.Conv2d(hw*4,hw*8,kernel_size=3,stride=1,padding=1,bias=True),
#                               norm(hw*8),nn.LeakyReLU(0.2,True),
#                               nn.Conv2d(hw*8,1,kernel_size=3,stride=1,padding=1),nn.Sigmoid()
#                               ]
#                   if self.use_sn:
#                         for i in range(len(model)):
#                             if isinstance(model[i], nn.Conv2d):
#                                 model[i] = spectral_norm(model[i])
#                   self.model=nn.Sequential(*model)
#           def forward(self,x):
#                     return self.model(x)
# 判别器结构patchGAN
# class Discriminator(nn.Module):
#           def __init__(self):
#                   super(Discriminator_T,self).__init__()
#                   self.model=[PatchGAN_D()]
#                   self.model=nn.Sequential(*self.model)

#           def forward(self,x):
#                 return self.model(x)
#           def init_train(self,grad=None):
#               for param in self.model.parameters():
#                   param.requires_grad = grad

# VGG各种权重下载地址
model_weight_urls = {
    "vgg11": "https://download.pytorch.org/models/vgg11-8a719046.pth",
    "vgg13": "https://download.pytorch.org/models/vgg13-19584684.pth",
    "vgg16": "https://download.pytorch.org/models/vgg16-397923af.pth",
    "vgg19": "https://download.pytorch.org/models/vgg19-dcbb9e9d.pth",
    "vgg11_bn": "https://download.pytorch.org/models/vgg11_bn-6002323d.pth",
    "vgg13_bn": "https://download.pytorch.org/models/vgg13_bn-abd245e5.pth",
    "vgg16_bn": "https://download.pytorch.org/models/vgg16_bn-6c64b313.pth",
    "vgg19_bn": "https://download.pytorch.org/models/vgg19_bn-c79401a0.pth",
}


# VGG19
class VGG19(nn.Module):
    def __init__(self, batch_norm=False, num_classes=1000):
        super(VGG19, self).__init__()
        self.cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512,
                    'M']
        self.batch_norm = batch_norm
        self.num_clases = num_classes
        self.features = self.make_layers(self.cfg, self.batch_norm)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        module_list = list(self.features.modules())
        for l in module_list[1:27]:  # conv4_4
            x = l(x)
        return x


# SL-LIN
class content_struct(nn.Module):
    def __init__(self):
        super(content_struct, self).__init__()
        #         self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        #         self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    #         self.fc1 = nn.Linear(128 * 16 * 16, 512)
    #         self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        #         x = nn.functional.relu(self.conv1(x))
        #         x = nn.functional.relu(self.conv2(x))
        x = self.pool(x)
        #         x = x.view(-1, 128 * 16 * 16)
        #         x = nn.functional.relu(self.fc1(x))
        #         x = self.fc2(x)
        return x

if __name__=="__main__":
    ge =GenEncoder()
    x= torch.rand(1,3,256,256)
    x = ge(x)
    print(x.shape)