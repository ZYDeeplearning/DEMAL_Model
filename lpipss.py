import argparse
import os
import lpips
import numpy as np
import torch
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-d0', '--dir0', type=str, default="120.40")
parser.add_argument('-d1', '--dir1', type=str, default="results/hayao/con")
parser.add_argument('-o', '--out', type=str, default='example_dists.txt')
parser.add_argument('-v', '--version', type=str, default='0.1')
parser.add_argument('--use_gpu', type=bool,default=True, help='turn on flag to use GPU')

opt = parser.parse_args()
# folder1 = "results\Mystyle\img"   # 第一个文件夹的实际路径
# folder2 = "results\Mystyle\sty"
## Initializing the model
loss_fn = lpips.LPIPS(net='alex', version=opt.version)
if (opt.use_gpu):
    loss_fn.cuda()

# crawl directories
f = open(opt.out, 'w')
files = os.listdir(opt.dir0)
lpips_values = []
for file in files:
    if (os.path.exists(os.path.join(opt.dir1, file))):

        # Load images
        img0 = lpips.im2tensor(lpips.load_image(os.path.join(opt.dir0, file)))  # RGB image from [-1,1]
        img1 = lpips.im2tensor(lpips.load_image(os.path.join(opt.dir1, file)))

        if (opt.use_gpu):
            img0 = img0.cuda()
            img1 = img1.cuda()
        # Compute distance
        dist01 = loss_fn.forward(img0, img1)
        lpips_values.append(dist01.cpu().detach().numpy())
        print('%s: %.3f' % (file, dist01))
        f.writelines('%s: %.6f\n' % (file, dist01))

f.close()
average_lpips = np.mean(lpips_values)
print(f'Average LPIPS: {average_lpips}')
