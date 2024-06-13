import torch
from thop import profile
from model import *

# Model
print('==> Building model..')
#

# xx= torch.rand(1,256,64,64)
# y = torch.rand(1, 1, 256, 256)
# z=torch.randn(1,512,16,16)
# #DEMAL model......................................................ours
# model = Generator()
# sty=GenEncoder()
# dis1 = Discriminator_S()
# dis2 = Discriminator_T()
# atten=attention()
# # atten=utm()
# flops1, params1 = profile(model, inputs=(x,))
# # flops4, params4 = profile(atten, inputs=(xx,xx))
# flops4, params4 = profile(atten, inputs=(xx,xx))
# flops5, params5 = profile(sty, inputs=(x,))
# flops2, params2 = profile(dis1, inputs=(x,))
# flops3, params3 = profile(dis2, inputs=(y,))
# # flops2, params2 = profile(sty, inputs=(x,))
# # flops3, params3 = profile(sm, inputs=(y,y))
# # flops4, params4 = profile(decoder, inputs=(y,))
#
# flops = flops1+flops2+flops3+flops4+flops5
# params = params1 + params2+ params3+params4+params5
#
# print('flops: ', flops, 'params: ', params)
# print('flops: %.2f M, params: %.2f M' % (flops/ 1000000.0, params / 1000000.0))
#CycleGAN-----------------------------------------------------------------------CycleGAN
# from baseline.cyclegan.model import *
# model = Generator(3, 32, 3, 6)
# model1 = Generator(3, 32, 3, 6)
# dis1 = Discriminator(3, 64, 1)
# dis2 = Discriminator(3, 64, 1)
# flops1, params1 = profile(model, inputs=(x,))
# flops2, params2 = profile(model1, inputs=(x,))
# flops3, params3 = profile(dis1, inputs=(x,))
# flops4, params4 = profile(dis2, inputs=(x,))
# flops = flops1+flops2+flops3+flops4
# params = params1 + params2+params3 + params4
#
# print('flops: ', flops, 'params: ', params)
# print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))

#CartoonGAN-----------------------------------------------------------------------CartoonGAN
# from baseline.cartoon.model import *
# model = generator(3, 3, 64,8)
# dis = discriminator(3, 3,32)
# flops1, params1 = profile(model, inputs=(x,))
# flops2, params2 = profile(dis, inputs=(x,))
# flops = flops1+flops2
# params = params1 + params2
#
# print('flops: ', flops, 'params: ', params)
# print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))

#AnimeGAN-----------------------------------------------------------------------AnimeGAN
# from baseline.animegan.model import *
# model = Generator('aaa')
# dis = Discriminator()
# with torch.autograd.profiler.profile(use_cuda=True, profile_memory=True) as prof:
#     out = model(x)
# print(prof)
# prof.export_chrome_trace('profiles')
# flops1, params1 = profile(model, inputs=(x,))
# flops2, params2 = profile(dis, inputs=(x,))
# flops = flops1+flops2
# params = params1 + params2
#
# print('flops: ', flops, 'params: ', params)
# print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))

#AnimeGAN-----------------------------------------------------------------------Whitebox
# from baseline.white.model import *
# model = Generator().to('cuda')
# dis = Discriminator()
# dis1 = Discriminator()
# for i in range(10):
#     x = torch.rand(1, 3, 512, 512).to('cuda')
#     with torch.autograd.profiler.profile(use_cuda=True, profile_memory=True) as prof:
#         out = model(x)
#     print(prof)
#     prof.export_chrome_trace('profiles')
# flops1, params1 = profile(model, inputs=(x,))
# flops2, params2 = profile(dis, inputs=(x,))
# flops3, params3 = profile(dis1, inputs=(x,))
# flops = flops1+flops2+flops3
# params = params1 + params2+params3

# print('flops: ', flops, 'params: ', params)
# print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))
#AnimeGAN-----------------------------------------------------------------------MScartoonGAN
# from baseline.MSCartoon.model import *
# encoder = GenEncoder()
# decoder = GenDecoder()
# dis =Discrimintor()
# aex = aux_classfier()
# flops1, params1 = profile(encoder, inputs=(x,))
# flops2, params2 = profile(decoder, inputs=(xx,))
# flops3, params3 = profile(dis, inputs=(x,))
# flops4, params4 = profile(aex, inputs=(x,))
#
# flops = flops1+flops2+flops3+flops4
# params = params1 + params2+params3+params4
# print('flops: ', flops, 'params: ', params)
# print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))
# #AnimeGAN-----------------------------------------------------------------------cartoonlossgan
# from baseline.cartoonloss.model import *
# xx = torch.rand(1, 128, 64, 64)
# d1 =AnimeDiscriminatorMy5(out_type='output').to('cuda')
# d = AnimeDiscriminatorMy5().to('cuda')
# g=AnimeGeneratorMy5().to('cuda')
# for i in range(10):
#     x = torch.randn(1,3,512,512).to('cuda')
#     with torch.autograd.profiler.profile(use_cuda=True, profile_memory=True) as prof:
#         out=g(d(x))
#     print(prof)
#     prof.export_chrome_trace('profiles')
# flops1, params1 = profile(d, inputs=(x,))
# flops2, params2 = profile(g, inputs=(xx,))
# flops3, params3 = profile(d1, inputs=(x,))
# flops = flops1+flops2+flops3
# params = params1 + params2+params3
#
# print('flops: ', flops, 'params: ', params)
# print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))

# #AnimeGAN-----------------------------------------------------------------------CCST
# from baseline.ccst.model import *
# model = G_net_unet().to('cuda')
# dis = D_net()
# dis1 = patch_D_net()
# for i in range(10):
#     x = torch.randn(1,3,512,512).to('cuda')
#     with torch.autograd.profiler.profile(use_cuda=True, profile_memory=True) as prof:
#         out=model(x)
#     print(prof)
#     prof.export_chrome_trace('profiles')
# flops1, params1 = profile(model, inputs=(x,))
# flops2, params2 = profile(dis, inputs=(x,))
# flops3, params3 = profile(dis1, inputs=(x,))
# flops = flops1+flops2+flops3
# params = params1 + params2+params3
# print('flops: ', flops, 'params: ', params)
# print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))
#AnimeGAN-----------------------------------------------------------------------CCPL
# from baseline.ccpl.model import *
# encoder = vgg
# model = vggcoder(encoder)
# sct = SCT()
# decoder = decoder
# decoder = Decoder(decoder)
#
# flops1, params1 = profile(model, inputs=(x,))
# flops2, params2 = profile(sct, inputs=(z,z))
# flops3, params3 = profile(decoder, inputs=(z,))
# flops = flops1+flops2+flops3
# params = params1 + params2+params3
# print('flops: ', flops, 'params: ', params)
# print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))
#AnimeGAN--------------------------------------------------------------------------------StyA2k
from baseline.StyA2K.model import *
encoder = vgg
model = vggcoder(encoder).to('cuda')
sct = A2Ks().to('cuda')
decoder = decoder
decoder = Decoder(decoder).to('cuda')
res={}
for i in range(10):
    x = torch.rand(1, 3, 512, 512).to('cuda')
    v = torch.rand(1, 3, 512, 512).to('cuda')
    y = torch.rand(1, 512, 32, 32).to('cuda')
    y1 = torch.rand(1, 512, 64, 64).to('cuda')
    y2 = torch.rand(1, 512, 128, 128).to('cuda')
    with torch.autograd.profiler.profile(use_cuda=True, profile_memory=True) as prof:
        x = model(x)
        v = model(v)
        res['cwct_3'] = sct(y2,y2,layer=3)
        res['cwct_2'] = sct(y1,y1, layer=4)
        x1 = sct(y, y, layer=5)
        out= decoder(y)
    print(prof)
    prof.export_chrome_trace('profiles')
flops1, params1 = profile(model, inputs=(x,))
flops2, params2 = profile(sct, inputs=(z,z,3))
flops22, params22 = profile(sct, inputs=(z,z,4))
flops222, params222 = profile(sct, inputs=(z,z,5))
flops3, params3 = profile(decoder, inputs=(z,))
flops = flops1+flops2+flops3+flops22+flops222
params = params1 + params2+params3+params22+params222
print('flops: ', flops, 'params: ', params)
print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))