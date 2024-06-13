import torch
from torch import nn
import numpy as np
import cv2
import os
from torch.autograd import Variable
import numpy as np
import cv2


def read_img(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    assert len(img.shape) == 3
    return img


# Calculates the average brightness in the specified irregular image
def calculate_average_brightness(img):
    # Average value of three color channels
    R = img[..., 0].mean()
    G = img[..., 1].mean()
    B = img[..., 2].mean()

    brightness = 0.299 * R + 0.587 * G + 0.114 * B
    return brightness, B, G, R


# adjusting the average brightness of the target image to the average brightness of the source image
def adjust_brightness_from_src_to_dst(dst, src, path=None, if_show=None, if_info=None):
    brightness1, B1, G1, R1 = calculate_average_brightness(src)
    brightness2, B2, G2, R2 = calculate_average_brightness(dst)
    brightness_difference = brightness1 / brightness2

    if if_info:
        print('Average brightness of original image', brightness1)
        print('Average brightness of target', brightness2)
        print('Brightness Difference between Original Image and Target', brightness_difference)

    dstf = dst * brightness_difference
    dstf = np.clip(dstf, 0, 255)
    dstf = np.uint8(dstf)

    ma, na, _ = src.shape
    mb, nb, _ = dst.shape
    result_show_img = np.zeros((max(ma, mb), 3 * max(na, nb), 3))
    result_show_img[:mb, :nb, :] = dst
    result_show_img[:ma, nb:nb + na, :] = src
    result_show_img[:mb, nb + na:nb + na + nb, :] = dstf
    result_show_img = result_show_img.astype(np.uint8)

    if if_show:
        cv2.imshow('-', cv2.cvtColor(result_show_img, cv2.COLOR_BGR2RGB))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if path != None:
        cv2.imwrite(path, cv2.cvtColor(result_show_img, cv2.COLOR_BGR2RGB))

    return dstf



# 参数权重初始化
def init_weights(m, init_type='normal', gain=0.02):
    classname = m.__class__.__name__
    if classname.find('BatchNorm2d') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            torch.nn.init.normal_(m.weight.data, 1.0, gain)
        if hasattr(m, 'bias') and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
        if init_type == 'normal':
            torch.nn.init.normal_(m.weight.data, 0.0, gain)
        elif init_type == 'xavier':
            torch.nn.init.xavier_normal_(m.weight.data, gain=gain)
        elif init_type == 'xavier_uniform':
            torch.nn.init.xavier_uniform_(m.weight.data, gain=1.0)
        elif init_type == 'kaiming':
            torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        elif init_type == 'orthogonal':
            torch.nn.init.orthogonal_(m.weight.data, gain=gain)
        elif init_type == 'none':  # uses pytorch's default init method
            m.reset_parameters()
        else:
            raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        if hasattr(m, 'bias') and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)


# 梯度
def requires_grad(model, flag=True):
    if model is None:
        return
    for p in model.parameters():
        p.requires_grad = flag

def adjust_contrast(img, contrast=0):
    # img: 0 ~ 255
    # value: -100 ~ 100

    img = img * 1.0
    thre = img.mean()
    img_out = img * 1.0
    if contrast <= -255.0:
        img_out = (img_out >= 0) + thre - 1
    elif contrast > -255.0 and contrast <= 0:
        img_out = img + (img - thre) * contrast / 255.0
    elif contrast <= 255.0 and contrast > 0:
        new_con = 255.0 * 255.0 / (256.0 - contrast) - 255.0
        img_out = img + (img - thre) * new_con / 255.0
    else:
        mask_1 = img > thre
        img_out = mask_1 * 255.0

    img_out = img_out / 255.0

    mask_1 = img_out < 0
    mask_2 = img_out > 1
    img_out = img_out * (1 - mask_1)
    img_out = img_out * (1 - mask_2) + mask_2

    img_out = img_out * 255.

    return img_out



def adjust_luminance(img, increment=0):

    # img: 0 ~ 255

    img = img * 1.0
    I = (img[:, :, 0] + img[:, :, 1] + img[:, :, 2]) / 3.0 + 0.001
    mask_1 = I > 128.0
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]
    rhs = (r * 128.0 - (I - 128.0) * 256.0) / (256.0 - I)
    ghs = (g * 128.0 - (I - 128.0) * 256.0) / (256.0 - I)
    bhs = (b * 128.0 - (I - 128.0) * 256.0) / (256.0 - I)
    rhs = rhs * mask_1 + (r * 128.0 / I) * (1 - mask_1)
    ghs = ghs * mask_1 + (g * 128.0 / I) * (1 - mask_1)
    bhs = bhs * mask_1 + (b * 128.0 / I) * (1 - mask_1)
    I_new = I + increment - 128.0
    mask_2 = I_new > 0.0
    R_new = rhs + (256.0 - rhs) * I_new / 128.0
    G_new = ghs + (256.0 - ghs) * I_new / 128.0
    B_new = bhs + (256.0 - bhs) * I_new / 128.0
    R_new = R_new * mask_2 + (rhs + rhs * I_new / 128.0) * (1 - mask_2)
    G_new = G_new * mask_2 + (ghs + ghs * I_new / 128.0) * (1 - mask_2)
    B_new = B_new * mask_2 + (bhs + bhs * I_new / 128.0) * (1 - mask_2)
    img_out = img * 1.0
    img_out[:, :, 0] = R_new
    img_out[:, :, 1] = G_new
    img_out[:, :, 2] = B_new
    img_out = img_out / 255.0

    mask_1 = img_out < 0
    mask_2 = img_out > 1
    img_out = img_out * (1 - mask_1)
    img_out = img_out * (1 - mask_2) + mask_2

    img_out = img_out * 255.

    return img_out


def adjust_saturation(img, increment=0):
    # img: 0 ~ 255
    # increment: -1 ~ 1

    img = img * 1.0
    img_out = img * 1.0

    img_min = img.min(axis=2)
    img_max = img.max(axis=2)

    Delta = (img_max - img_min) / 255.0
    value = (img_max + img_min) / 255.0
    L = value / 2.0

    mask_1 = L < 0.5

    s1 = Delta / (value + 0.001)
    s2 = Delta / (2 - value + 0.001)
    s = s1 * mask_1 + s2 * (1 - mask_1)

    if increment >= 0:
        temp = increment + s
        mask_2 = temp > 1
        alpha_1 = s
        alpha_2 = s * 0 + 1 - increment
        alpha = alpha_1 * mask_2 + alpha_2 * (1 - mask_2)
        alpha = 1 / (alpha + 0.001) - 1
        img_out[:, :, 0] = img[:, :, 0] + (img[:, :, 0] - L * 255.0) * alpha
        img_out[:, :, 1] = img[:, :, 1] + (img[:, :, 1] - L * 255.0) * alpha
        img_out[:, :, 2] = img[:, :, 2] + (img[:, :, 2] - L * 255.0) * alpha

    else:
        alpha = increment
        img_out[:, :, 0] = L * 255.0 + (img[:, :, 0] - L * 255.0) * (1 + alpha)
        img_out[:, :, 1] = L * 255.0 + (img[:, :, 1] - L * 255.0) * (1 + alpha)
        img_out[:, :, 2] = L * 255.0 + (img[:, :, 2] - L * 255.0) * (1 + alpha)

    img_out = img_out / 255.0

    mask_1 = img_out < 0
    mask_2 = img_out > 1

    img_out = img_out * (1 - mask_1)
    img_out = img_out * (1 - mask_2) + mask_2

    img_out = img_out * 255.

    return img_out

def inverse_transform(images):
    images = (images + 1.) / 2 * 255   # -1 ~ 1 --> 0 ~ 255
    images = np.clip(images, 0, 255)
    return images

def save_images(images, dataset_name, image_path, photo_path=None):
    images = inverse_transform(images.squeeze())
    adjust_config = None
    if dataset_name == 'hayao':
        adjust_config = [0.5, 50, 10]
    elif dataset_name == 'shinkai':
        adjust_config = [0.42, 70, 25]
    elif dataset_name == 'paprika':
        adjust_config = [0.35, 25, 20]
    else:
        raise ValueError('invalid dataset name.')
    if photo_path:
        images = adjust_brightness_from_src_to_dst(images,read_img(photo_path))
        images = adjust_saturation(images, adjust_config[0])
        images = adjust_contrast(images, adjust_config[1])
        images = adjust_luminance(images, adjust_config[2])
        images = images.astype(np.uint8)
        return imsave(images, image_path)

def imsave(images, path):
    return cv2.imwrite(path, cv2.cvtColor(images, cv2.COLOR_BGR2RGB))

def rgb_to_yuv(image, x):
    image = (image + 1.0) / 2.0
    yuv_img = torch.tensordot(
        image,
        x,
        dims=([image.ndim - 3], [0]))
    return yuv_img


# 时间转化
def time_change(time):
    new_time = t.localtime(time)
    new_time = t.strftime("%Hh%Mm%Ss", new_time)
    return new_time


# 归一化
def denorm(x):
    x = (x * 0.5 + 0.5) * 255.0
    return x.cpu().detach().numpy().transpose(1, 2, 0)


def process(x):
    x = (x * 0.5 + 0.5) * 255.0
    return x.cpu().detach().numpy()[0][0]


def RGB2BGR(x):
    return cv2.cvtColor(x, cv2.COLOR_RGB2BGR)


# 创建文件目录
def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


# 颜色RGB->Gray(DCT-NET)
def color_shift(image, mode='uniform'):
    device = image.device
    b1, g1, r1 = torch.split(image, 1, dim=1)
    if mode == 'normal':
        b_weight = torch.normal(mean=0.114, std=0.1, size=[1]).to(device)
        g_weight = torch.normal(mean=0.587, std=0.1, size=[1]).to(device)
        r_weight = torch.normal(mean=0.299, std=0.1, size=[1]).to(device)
    elif mode == 'uniform':
        b_weight = torch.FloatTensor(1).uniform_(0.014, 0.214).to(device)
        g_weight = torch.FloatTensor(1).uniform_(0.487, 0.687).to(device)
        r_weight = torch.FloatTensor(1).uniform_(0.199, 0.399).to(device)
    output1 = (b_weight * b1 + g_weight * g1 + r_weight * r1) / (b_weight + g_weight + r_weight)
    return output1


# 图像pool
class Queue():
    def __init__(self, pool_size=50):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images
        return_images = []
        for image in images.data:
            image = torch.unsqueeze(image, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size - 1)
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return_images = Variable(torch.cat(return_images, 0))
        return return_images


# 颜色white-box
class ColorShift():
    def __init__(self, device: torch.device = 'cuda', mode='uniform', image_format='rgb'):
        self.dist: torch.distributions = None
        self.dist_param1: torch.Tensor = None
        self.dist_param2: torch.Tensor = None

        if (mode == 'uniform'):
            self.dist_param1 = torch.tensor((0.199, 0.487, 0.014), device=device)
            self.dist_param2 = torch.tensor((0.399, 0.687, 0.214), device=device)
            if (image_format == 'bgr'):
                self.dist_param1 = torch.permute(self.dist_param1, (2, 1, 0))
                self.dist_param2 = torch.permute(self.dist_param2, (2, 1, 0))

            self.dist = torch.distributions.Uniform(low=self.dist_param1, high=self.dist_param2)


        elif (mode == 'normal'):
            self.dist_param1 = torch.tensor((0.299, 0.587, 0.114), device=device)
            self.dist_param2 = torch.tensor((0.1, 0.1, 0.1), device=device)
            if (image_format == 'bgr'):
                self.dist_param1 = torch.permute(self.dist_param1, (2, 1, 0))
                self.dist_param2 = torch.permute(self.dist_param2, (2, 1, 0))

            self.dist = torch.distributions.Normal(loc=self.dist_param1, scale=self.dist_param2)

    # Allow taking mutiple images batches as input
    # So we can do: gray_fake, gray_cartoon = ColorShift(output, input_cartoon)
    def process(self, *image_batches: torch.Tensor):
        # Sample the random color shift coefficients
        weights = self.dist.sample()

        # images * weights[None, :, None, None] => Apply weights to r,g,b channels of each images
        # torch.sum(, dim=1) => Sum along the channels so (B, 3, H, W) become (B, H, W)
        # .unsqueeze(1) => add back the channel so (B, H, W) become (B, 1, H, W)
        # .repeat(1, 3, 1, 1) => (B, 1, H, W) become (B, 3, H, W) again
        return (
            (((torch.sum(images * weights[None, :, None, None], dim=1)) / weights.sum()).unsqueeze(1)).repeat(1, 3, 1,
                                                                                                              1)
            for images in image_batches)


# patch 抽取
def extract_image_patches(x, kernel, stride):
    if kernel != 1:
        x = nn.ZeroPad2d(1)(x)
    x = x.permute(0, 2, 3, 1)
    all_patches = x.unfold(1, kernel, stride).unfold(2, kernel, stride)
    all_patches = all_patches.permute(0, 3, 1, 2, 4, 5)
    all_patches = all_patches.reshape([all_patches.shape[0] * all_patches.shape[2] ** 2, all_patches.shape[1],
                                       all_patches.shape[4], all_patches.shape[5]])
    return all_patches


def gram(input):
    b, c, w, h = input.size()
    x = input.view(b * c, w * h)
    G = torch.mm(x, x.T)
    return G.div(b * c * w * h)


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    return num_params


def high_pass_filter(img, d, n):
    return (1 - 1 / (1 + (img / d) ** n))


# 引导滤波
class GuidedFilter():
    def box_filter(self, x, r):
        channel = x.shape[1]  # Batch, Channel, H, W
        kernel_size = (2 * r + 1)
        weight = 1.0 / (kernel_size ** 2)
        box_kernel = weight * torch.ones((channel, 1, kernel_size, kernel_size), dtype=torch.float32, device=x.device)
        output = F.conv2d(x, weight=box_kernel, stride=1, padding=r,
                          groups=channel)  # tf.nn.depthwise_conv2d(x, box_kernel, [1, 1, 1, 1], 'SAME')
        return output

    def guided_filter(self, x, y, r, eps=1e-2):
        # Batch, Channel, H, W
        _, _, H, W = x.shape

        N = self.box_filter(torch.ones((1, 1, H, W), dtype=x.dtype, device=x.device), r)

        mean_x = self.box_filter(x, r) / N
        mean_y = self.box_filter(y, r) / N
        cov_xy = self.box_filter(x * y, r) / N - mean_x * mean_y
        var_x = self.box_filter(x * x, r) / N - mean_x * mean_x

        A = cov_xy / (var_x + eps)
        b = mean_y - A * mean_x

        mean_A = self.box_filter(A, r) / N
        mean_b = self.box_filter(b, r) / N

        output = mean_A * x + mean_b
        return output


# tv_loss
class VariationLoss(nn.Module):
    def __init__(self, k_size: int) -> None:
        super().__init__()
        self.k_size = k_size

    def forward(self, image: torch.Tensor):
        b, c, h, w = image.shape
        tv_h = torch.mean((image[:, :, self.k_size:, :] - image[:, :, : -self.k_size, :]) ** 2)
        tv_w = torch.mean((image[:, :, :, self.k_size:] - image[:, :, :, : -self.k_size]) ** 2)
        tv_loss = (tv_h + tv_w) / (3 * h * w)
        return tv_loss


def get_gaussian_kernel(k=3, mu=0, sigma=1, normalize=True):
    # compute 1 dimension gaussian
    gaussian_1D = np.linspace(-1, 1, k)
    # compute a grid distance from center
    x, y = np.meshgrid(gaussian_1D, gaussian_1D)
    distance = (x ** 2 + y ** 2) ** 0.5

    # compute the 2 dimension gaussian
    gaussian_2D = np.exp(-(distance - mu) ** 2 / (2 * sigma ** 2))
    gaussian_2D = gaussian_2D / (2 * np.pi * sigma ** 2)

    # normalize part (mathematically)
    if normalize:
        gaussian_2D = gaussian_2D / np.sum(gaussian_2D)
    return gaussian_2D


def get_sobel_kernel(k=3):
    # get range
    range = np.linspace(-(k // 2), k // 2, k)
    # compute a grid the numerator and the axis-distances
    x, y = np.meshgrid(range, range)
    sobel_2D_numerator = x
    sobel_2D_denominator = (x ** 2 + y ** 2)
    sobel_2D_denominator[:, k // 2] = 1  # avoid division by zero
    sobel_2D = sobel_2D_numerator / sobel_2D_denominator
    return sobel_2D


def get_thin_kernels(start=0, end=360, step=45):
    k_thin = 3  # actual size of the directional kernel
    # increase for a while to avoid interpolation when rotating
    k_increased = k_thin + 2

    # get 0° angle directional kernel
    thin_kernel_0 = np.zeros((k_increased, k_increased))
    thin_kernel_0[k_increased // 2, k_increased // 2] = 1
    thin_kernel_0[k_increased // 2, k_increased // 2 + 1:] = -1

    # rotate the 0° angle directional kernel to get the other ones
    thin_kernels = []
    for angle in range(start, end, step):
        (h, w) = thin_kernel_0.shape
        # get the center to not rotate around the (0, 0) coord point
        center = (w // 2, h // 2)
        # apply rotation
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
        kernel_angle_increased = cv2.warpAffine(thin_kernel_0, rotation_matrix, (w, h), cv2.INTER_NEAREST)

        # get the k=3 kerne
        kernel_angle = kernel_angle_increased[1:-1, 1:-1]
        is_diag = (abs(kernel_angle) == 1)  # because of the interpolation
        kernel_angle = kernel_angle * is_diag  # because of the interpolation
        thin_kernels.append(kernel_angle)
    return thin_kernels


# Canny 边缘检测
class CannyFilter(nn.Module):
    def __init__(self,
                 k_gaussian=3,
                 mu=0,
                 sigma=1,
                 k_sobel=3,
                 device='cuda:0'):
        super(CannyFilter, self).__init__()
        # device
        self.device = device
        # gaussian
        gaussian_2D = get_gaussian_kernel(k_gaussian, mu, sigma)
        self.gaussian_filter = nn.Conv2d(in_channels=1,
                                         out_channels=1,
                                         kernel_size=k_gaussian,
                                         padding=k_gaussian // 2,
                                         bias=False)
        self.gaussian_filter.weight.data[:, :] = nn.Parameter(torch.from_numpy(gaussian_2D), requires_grad=False)

        # sobel

        sobel_2D = get_sobel_kernel(k_sobel)
        self.sobel_filter_x = nn.Conv2d(in_channels=1,
                                        out_channels=1,
                                        kernel_size=k_sobel,
                                        padding=k_sobel // 2,
                                        bias=False)
        self.sobel_filter_x.weight.data[:, :] = nn.Parameter(torch.from_numpy(sobel_2D), requires_grad=False)

        self.sobel_filter_y = nn.Conv2d(in_channels=1,
                                        out_channels=1,
                                        kernel_size=k_sobel,
                                        padding=k_sobel // 2,
                                        bias=False)
        self.sobel_filter_y.weight.data[:, :] = nn.Parameter(torch.from_numpy(sobel_2D.T), requires_grad=False)

        # thin

        thin_kernels = get_thin_kernels()
        directional_kernels = np.stack(thin_kernels)

        self.directional_filter = nn.Conv2d(in_channels=1,
                                            out_channels=8,
                                            kernel_size=thin_kernels[0].shape,
                                            padding=thin_kernels[0].shape[-1] // 2,
                                            bias=False)
        self.directional_filter.weight.data[:, 0] = nn.Parameter(torch.from_numpy(directional_kernels),
                                                                 requires_grad=False)

        # hysteresis

        hysteresis = np.ones((3, 3)) + 0.25
        self.hysteresis = nn.Conv2d(in_channels=1,
                                    out_channels=1,
                                    kernel_size=3,
                                    padding=1,
                                    bias=False)
        self.hysteresis.weight.data[:, :] = nn.Parameter(torch.from_numpy(hysteresis), requires_grad=False)

    def forward(self, img, low_threshold=None, high_threshold=None, hysteresis=True):
        # set the setps tensors
        B, C, H, W = img.shape
        blurred = torch.zeros((B, C, H, W)).to(self.device)
        grad_x = torch.zeros((B, 1, H, W)).to(self.device)
        grad_y = torch.zeros((B, 1, H, W)).to(self.device)
        grad_magnitude = torch.zeros((B, 1, H, W)).to(self.device)
        grad_orientation = torch.zeros((B, 1, H, W)).to(self.device)

        # gaussian

        for c in range(C):
            blurred[:, c:c + 1] = self.gaussian_filter(img[:, c:c + 1])
            grad_x = grad_x + self.sobel_filter_x(blurred[:, c:c + 1])
            grad_y = grad_y + self.sobel_filter_y(blurred[:, c:c + 1])

        # thick edges

        grad_x, grad_y = grad_x / C, grad_y / C
        grad_magnitude = (grad_x ** 2 + grad_y ** 2) ** 0.5
        grad_orientation = torch.atan2(grad_y, grad_x)
        grad_orientation = grad_orientation * (180 / np.pi) + 180  # convert to degree
        grad_orientation = torch.round(grad_orientation / 45) * 45  # keep a split by 45

        # thin edges

        directional = self.directional_filter(grad_magnitude)
        # get indices of positive and negative directions
        positive_idx = (grad_orientation / 45) % 8
        negative_idx = ((grad_orientation / 45) + 4) % 8
        thin_edges = grad_magnitude.clone()
        # non maximum suppression direction by direction
        for pos_i in range(4):
            neg_i = pos_i + 4
            # get the oriented grad for the angle
            is_oriented_i = (positive_idx == pos_i) * 1
            is_oriented_i = is_oriented_i + (positive_idx == neg_i) * 1
            pos_directional = directional[:, pos_i]
            neg_directional = directional[:, neg_i]
            selected_direction = torch.stack([pos_directional, neg_directional])

            # get the local maximum pixels for the angle
            # selected_direction.min(dim=0)返回一个列表[0]中包含两者中的小的，[1]包含了小值的索引
            is_max = selected_direction.min(dim=0)[0] > 0.0
            is_max = torch.unsqueeze(is_max, dim=1)

            # apply non maximum suppression
            to_remove = (is_max == 0) * 1 * (is_oriented_i) > 0
            thin_edges[to_remove] = 0.0

        # thresholds

        if low_threshold is not None:
            low = thin_edges > low_threshold

            if high_threshold is not None:
                high = thin_edges > high_threshold
                # get black/gray/white only
                thin_edges = low * 0.5 + high * 0.5

                if hysteresis:
                    # get weaks and check if they are high or not
                    weak = (thin_edges == 0.5) * 1
                    weak_is_high = (self.hysteresis(thin_edges) > 1) * weak
                    thin_edges = high * 1 + weak_is_high * 1
            else:
                thin_edges = low * 1

        return thin_edges * 255


# 掩码操作mask
def mask(img, mask, y1, y2, Canny):
    edge_fake_img = Canny(img, y1, y2)
    edge_real_img = Canny(mask, y1, y2)
    edge_fake_img = high_pass_filter(edge_fake_img, d=0.2, n=2)
    edge_real_img = high_pass_filter(edge_real_img, d=0.2, n=2)
    # 转换
    edge_real_img = process(edge_real_img)
    edge_fake_img = process(edge_fake_img)
    result = cv2.bitwise_and(edge_fake_img, edge_real_img)
    result = torch.from_numpy(result)
    edge_real_img = torch.from_numpy(edge_real_img)
    return result, edge_real_img


# 计算均值
def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def calc_mean(feat):
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean


def nor_mean_std(feat):
    size = feat.size()
    mean, std = calc_mean_std(feat)
    nor_feat = (feat - mean.expand(size)) / std.expand(size)
    return nor_feat


def nor_mean(feat):
    size = feat.size()
    mean = calc_mean(feat)
    nor_feat = feat - mean.expand(size)
    return nor_feat, mean


def calc_cov(feat):
    feat = feat.flatten(2, 3)
    f_cov = torch.bmm(feat, feat.permute(0, 2, 1)).div(feat.size(2))
    return f_cov


if __name__=='__main__':
    x = torch.rand(1,32,64,64)
    out = calc_cov(x)
    print(out.shape)
