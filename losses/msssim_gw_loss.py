import torch
import math
import torch.nn.functional as F
from torch.autograd import Variable


class GMSSSIMLoss(torch.nn.Module):
    def __init__(self, loss_weight=1.0, size_average=True, max_val=1, device='cuda'):
        super(GMSSSIMLoss, self).__init__()
        self.size_average = size_average
        self.max_val = max_val
        self.device = device
        self.weight = torch.tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], device=device)
        self.loss_weight = loss_weight
        sobel_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        sobel_y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
        self.sobel_x = torch.FloatTensor(sobel_x).to(device)
        self.sobel_y = torch.FloatTensor(sobel_y).to(device)

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()

    def create_window(self, window_size, sigma, channel):
        _1D_window = self.gaussian(window_size, sigma).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = Variable(_2D_window.expand(channel, 1, window_size, window_size)).contiguous()
        return window

    def _ssim(self, img1, img2, size_average = True):
        _, c, w, h = img1.size()
        window_size = min(w, h, 11)
        sigma = 1.5 * window_size / 11
        window = self.create_window(window_size, sigma, c).to(self.device)
        mu1 = F.conv2d(img1, window, padding=window_size//2, groups=c)
        mu2 = F.conv2d(img2, window, padding=window_size//2, groups=c)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1*mu2

        weight_x = self.sobel_x.expand(c, 1, 3, 3)
        weight_y = self.sobel_y.expand(c, 1, 3, 3)

        Ix1 = torch.abs(F.conv2d(img1, weight_x, stride=1, padding=1, groups=c))
        Iy1 = torch.abs(F.conv2d(img1, weight_y, stride=1, padding=1, groups=c))
        Ix2 = torch.abs(F.conv2d(img2, weight_x, stride=1, padding=1, groups=c))
        Iy2 = torch.abs(F.conv2d(img2, weight_y, stride=1, padding=1, groups=c))

        gradientimg1 = Ix1 + Iy1
        gradientimg2 = Ix2 + Iy2

        gradientimg1_2 = gradientimg1 * gradientimg1
        gradientimg2_2 = gradientimg2 * gradientimg2
        gradientimg1_gradientimg2 = gradientimg1 * gradientimg2

        mu_gimg1 = F.conv2d(gradientimg1, window, padding=window_size//2, groups=c)
        mu_gimg2 = F.conv2d(gradientimg2, window, padding=window_size//2, groups=c)
        mu_gimg1_2 = mu_gimg1 * mu_gimg1
        mu_gimg2_2 = mu_gimg2 * mu_gimg2
        mu_gimg1_gimg2 = mu_gimg1 * mu_gimg2

        sigma_gimg1_2 = F.conv2d(gradientimg1_2, window, padding=window_size//2, groups=c) - mu_gimg1_2
        sigma_gimg2_2 = F.conv2d(gradientimg2_2, window, padding=window_size//2, groups=c) - mu_gimg2_2
        sigma_gimg1_gimg2 = F.conv2d(gradientimg1_gradientimg2, window, padding=window_size//2, groups=c) - mu_gimg1_gimg2

        # sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=c) - mu1_sq
        # sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=c) - mu2_sq
        # sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=c) - mu1_mu2

        C1 = (0.01 * self.max_val) **2
        C2 = (0.03 * self.max_val) **2
        # V1 = 2.0 * sigma12 + C2
        # V2 = sigma1_sq + sigma2_sq + C2
        V1 = 2.0 * sigma_gimg1_gimg2 + C2
        V2 = sigma_gimg1_2 + sigma_gimg2_2 + C2
        ssim_map = ((2 * mu1_mu2 + C1) * V1) / ((mu1_sq + mu2_sq + C1) * V2)
        mcs_map = V1 / V2
        if size_average:
            return ssim_map.mean(), mcs_map.mean()

    def ms_ssim(self, img1, img2, levels=5):
        msssim = torch.empty(levels, device=self.device)
        mcs = torch.empty(levels, device=self.device)
        for i in range(levels):
            ssim_map, mcs_map = self._ssim(img1, img2)
            # msssim[i] = ssim_map
            # mcs[i] = mcs_map
            msssim[i] = torch.relu(ssim_map)
            mcs[i] = torch.relu(mcs_map)
            filtered_im1 = F.avg_pool2d(img1, kernel_size=2, stride=2)
            filtered_im2 = F.avg_pool2d(img2, kernel_size=2, stride=2)
            img1 = filtered_im1
            img2 = filtered_im2

        value = (torch.prod(mcs[0:levels-1] ** self.weight[0:levels-1])*
                                    (msssim[levels-1] ** self.weight[levels-1]))
        return value


    def forward(self, img1, img2):
        img1 = torch.relu(img1)
        img2 = torch.relu(img2)
        return self.loss_weight*(1-self.ms_ssim(img1, img2))


class MSSSIMGWLoss(torch.nn.Module):
    def __init__(self, loss_weight=1.0, size_average=True, max_val=1, device='cuda'):
        super(MSSSIMGWLoss, self).__init__()
        self.size_average = size_average
        self.max_val = max_val
        self.device = device
        self.weight = torch.tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], device=device)
        self.loss_weight = loss_weight
        sobel_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        sobel_y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
        self.sobel_x = torch.FloatTensor(sobel_x).to(device)
        self.sobel_y = torch.FloatTensor(sobel_y).to(device)

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()

    def create_window(self, window_size, sigma, channel):
        _1D_window = self.gaussian(window_size, sigma).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = Variable(_2D_window.expand(channel, 1, window_size, window_size)).contiguous()
        return window

    def _ssim(self, img1, img2, size_average = True):
        _, c, w, h = img1.size()
        window_size = min(w, h, 11)
        sigma = 1.5 * window_size / 11
        window = self.create_window(window_size, sigma, c).to(self.device)
        mu1 = F.conv2d(img1, window, padding=window_size//2, groups=c)
        mu2 = F.conv2d(img2, window, padding=window_size//2, groups=c)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1*mu2

        weight_x = self.sobel_x.expand(c, 1, 3, 3)
        weight_y = self.sobel_y.expand(c, 1, 3, 3)

        Ix1 = F.conv2d(img1, weight_x, stride=1, padding=1, groups=c)
        Iy1 = F.conv2d(img1, weight_y, stride=1, padding=1, groups=c)
        Ix2 = F.conv2d(img2, weight_x, stride=1, padding=1, groups=c)
        Iy2 = F.conv2d(img2, weight_y, stride=1, padding=1, groups=c)

        gradientimg1 = Ix1 + Iy1
        gradientimg2 = Ix2 + Iy2

        gradientimg1_2 = gradientimg1 * gradientimg1
        gradientimg2_2 = gradientimg2 * gradientimg2
        gradientimg1_gradientimg2 = gradientimg1 * gradientimg2

        mu_gimg1 = F.conv2d(gradientimg1, window, padding=window_size//2, groups=c)
        mu_gimg2 = F.conv2d(gradientimg2, window, padding=window_size//2, groups=c)
        mu_gimg1_2 = mu_gimg1 * mu_gimg1
        mu_gimg2_2 = mu_gimg2 * mu_gimg2
        mu_gimg1_gimg2 = mu_gimg1 * mu_gimg2

        sigma_gimg1_2 = F.conv2d(gradientimg1_2, window, padding=window_size//2, groups=c) - mu_gimg1_2
        sigma_gimg2_2 = F.conv2d(gradientimg2_2, window, padding=window_size//2, groups=c) - mu_gimg2_2
        sigma_gimg1_gimg2 = F.conv2d(gradientimg1_gradientimg2, window, padding=window_size//2, groups=c) - mu_gimg1_gimg2

        # sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=c) - mu1_sq
        # sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=c) - mu2_sq
        # sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=c) - mu1_mu2

        C1 = (0.01 * self.max_val) **2
        C2 = (0.03 * self.max_val) **2
        # V1 = 2.0 * sigma12 + C2
        # V2 = sigma1_sq + sigma2_sq + C2
        V1 = 2.0 * sigma_gimg1_gimg2 + C2
        V2 = sigma_gimg1_2 + sigma_gimg2_2 + C2
        ssim_map = ((2 * mu1_mu2 + C1) * V1) / ((mu1_sq + mu2_sq + C1) * V2)
        mcs_map = V1 / V2
        if size_average:
            return ssim_map.mean(), mcs_map.mean()

    def ms_ssim(self, img1, img2, levels=5):
        msssim = torch.empty(levels, device=self.device)
        mcs = torch.empty(levels, device=self.device)
        for i in range(levels):
            ssim_map, mcs_map = self._ssim(img1, img2)
            # msssim[i] = ssim_map
            # mcs[i] = mcs_map
            msssim[i] = torch.relu(ssim_map)
            mcs[i] = torch.relu(mcs_map)
            filtered_im1 = F.avg_pool2d(img1, kernel_size=2, stride=2)
            filtered_im2 = F.avg_pool2d(img2, kernel_size=2, stride=2)
            img1 = filtered_im1
            img2 = filtered_im2

        value = (torch.prod(mcs[0:levels-1] ** self.weight[0:levels-1])*
                                    (msssim[levels-1] ** self.weight[levels-1]))
        return value


    def forward(self, img1, img2):
        img1 = torch.relu(img1)
        img2 = torch.relu(img2)
        return self.loss_weight*(1-self.ms_ssim(img1, img2))


class MSSSIMGWLossX(torch.nn.Module):
    def __init__(self, loss_weight=1.0, size_average=True, max_val=1, device='cuda'):
        super(MSSSIMGWLossX, self).__init__()
        self.size_average = size_average
        self.max_val = max_val
        self.device = device
        self.weight = torch.tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], device=device)
        self.loss_weight = loss_weight
        sobel_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        self.sobel_x = torch.FloatTensor(sobel_x).to(device)

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()

    def create_window(self, window_size, sigma, channel):
        _1D_window = self.gaussian(window_size, sigma).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = Variable(_2D_window.expand(channel, 1, window_size, window_size)).contiguous()
        return window

    def _ssim(self, img1, img2, size_average = True):
        _, c, w, h = img1.size()
        window_size = min(w, h, 11)
        sigma = 1.5 * window_size / 11
        window = self.create_window(window_size, sigma, c).to(self.device)
        mu1 = F.conv2d(img1, window, padding=window_size//2, groups=c)
        mu2 = F.conv2d(img2, window, padding=window_size//2, groups=c)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1*mu2

        weight_x = self.sobel_x.expand(c, 1, 3, 3)

        Ix1 = F.conv2d(img1, weight_x, stride=1, padding=1, groups=c)
        Ix2 = F.conv2d(img2, weight_x, stride=1, padding=1, groups=c)

        gradientimg1 = Ix1
        gradientimg2 = Ix2

        gradientimg1_2 = gradientimg1 * gradientimg1
        gradientimg2_2 = gradientimg2 * gradientimg2
        gradientimg1_gradientimg2 = gradientimg1 * gradientimg2

        mu_gimg1 = F.conv2d(gradientimg1, window, padding=window_size//2, groups=c)
        mu_gimg2 = F.conv2d(gradientimg2, window, padding=window_size//2, groups=c)
        mu_gimg1_2 = mu_gimg1 * mu_gimg1
        mu_gimg2_2 = mu_gimg2 * mu_gimg2
        mu_gimg1_gimg2 = mu_gimg1 * mu_gimg2

        sigma_gimg1_2 = F.conv2d(gradientimg1_2, window, padding=window_size//2, groups=c) - mu_gimg1_2
        sigma_gimg2_2 = F.conv2d(gradientimg2_2, window, padding=window_size//2, groups=c) - mu_gimg2_2
        sigma_gimg1_gimg2 = F.conv2d(gradientimg1_gradientimg2, window, padding=window_size//2, groups=c) - mu_gimg1_gimg2

        # sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=c) - mu1_sq
        # sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=c) - mu2_sq
        # sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=c) - mu1_mu2

        C1 = (0.01 * self.max_val) **2
        C2 = (0.03 * self.max_val) **2
        # V1 = 2.0 * sigma12 + C2
        # V2 = sigma1_sq + sigma2_sq + C2
        V1 = 2.0 * sigma_gimg1_gimg2 + C2
        V2 = sigma_gimg1_2 + sigma_gimg2_2 + C2
        ssim_map = ((2 * mu1_mu2 + C1) * V1) / ((mu1_sq + mu2_sq + C1) * V2)
        mcs_map = V1 / V2
        if size_average:
            return ssim_map.mean(), mcs_map.mean()

    def ms_ssim(self, img1, img2, levels=5):
        msssim = torch.empty(levels, device=self.device)
        mcs = torch.empty(levels, device=self.device)
        for i in range(levels):
            ssim_map, mcs_map = self._ssim(img1, img2)
            # msssim[i] = ssim_map
            # mcs[i] = mcs_map
            msssim[i] = torch.relu(ssim_map)
            mcs[i] = torch.relu(mcs_map)
            filtered_im1 = F.avg_pool2d(img1, kernel_size=2, stride=2)
            filtered_im2 = F.avg_pool2d(img2, kernel_size=2, stride=2)
            img1 = filtered_im1
            img2 = filtered_im2

        value = (torch.prod(mcs[0:levels-1] ** self.weight[0:levels-1])*
                                    (msssim[levels-1] ** self.weight[levels-1]))
        return value


    def forward(self, img1, img2):
        img1 = torch.relu(img1)
        img2 = torch.relu(img2)
        return self.loss_weight*(1-self.ms_ssim(img1, img2))


class MSSSIMGWLossY(torch.nn.Module):
    def __init__(self, loss_weight=1.0, size_average=True, max_val=1, device='cuda'):
        super(MSSSIMGWLossY, self).__init__()
        self.size_average = size_average
        self.max_val = max_val
        self.device = device
        self.weight = torch.tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], device=device)
        self.loss_weight = loss_weight
        sobel_y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
        self.sobel_y = torch.FloatTensor(sobel_y).to(device)

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()

    def create_window(self, window_size, sigma, channel):
        _1D_window = self.gaussian(window_size, sigma).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = Variable(_2D_window.expand(channel, 1, window_size, window_size)).contiguous()
        return window

    def _ssim(self, img1, img2, size_average = True):
        _, c, w, h = img1.size()
        window_size = min(w, h, 11)
        sigma = 1.5 * window_size / 11
        window = self.create_window(window_size, sigma, c).to(self.device)
        mu1 = F.conv2d(img1, window, padding=window_size//2, groups=c)
        mu2 = F.conv2d(img2, window, padding=window_size//2, groups=c)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1*mu2

        weight_y = self.sobel_y.expand(c, 1, 3, 3)

        Iy1 = F.conv2d(img1, weight_y, stride=1, padding=1, groups=c)
        Iy2 = F.conv2d(img2, weight_y, stride=1, padding=1, groups=c)

        gradientimg1 = Iy1
        gradientimg2 = Iy2

        gradientimg1_2 = gradientimg1 * gradientimg1
        gradientimg2_2 = gradientimg2 * gradientimg2
        gradientimg1_gradientimg2 = gradientimg1 * gradientimg2

        mu_gimg1 = F.conv2d(gradientimg1, window, padding=window_size//2, groups=c)
        mu_gimg2 = F.conv2d(gradientimg2, window, padding=window_size//2, groups=c)
        mu_gimg1_2 = mu_gimg1 * mu_gimg1
        mu_gimg2_2 = mu_gimg2 * mu_gimg2
        mu_gimg1_gimg2 = mu_gimg1 * mu_gimg2

        sigma_gimg1_2 = F.conv2d(gradientimg1_2, window, padding=window_size//2, groups=c) - mu_gimg1_2
        sigma_gimg2_2 = F.conv2d(gradientimg2_2, window, padding=window_size//2, groups=c) - mu_gimg2_2
        sigma_gimg1_gimg2 = F.conv2d(gradientimg1_gradientimg2, window, padding=window_size//2, groups=c) - mu_gimg1_gimg2

        # sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=c) - mu1_sq
        # sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=c) - mu2_sq
        # sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=c) - mu1_mu2

        C1 = (0.01 * self.max_val) **2
        C2 = (0.03 * self.max_val) **2
        # V1 = 2.0 * sigma12 + C2
        # V2 = sigma1_sq + sigma2_sq + C2
        V1 = 2.0 * sigma_gimg1_gimg2 + C2
        V2 = sigma_gimg1_2 + sigma_gimg2_2 + C2
        ssim_map = ((2 * mu1_mu2 + C1) * V1) / ((mu1_sq + mu2_sq + C1) * V2)
        mcs_map = V1 / V2
        if size_average:
            return ssim_map.mean(), mcs_map.mean()

    def ms_ssim(self, img1, img2, levels=5):
        msssim = torch.empty(levels, device=self.device)
        mcs = torch.empty(levels, device=self.device)
        for i in range(levels):
            ssim_map, mcs_map = self._ssim(img1, img2)
            # msssim[i] = ssim_map
            # mcs[i] = mcs_map
            msssim[i] = torch.relu(ssim_map)
            mcs[i] = torch.relu(mcs_map)
            filtered_im1 = F.avg_pool2d(img1, kernel_size=2, stride=2)
            filtered_im2 = F.avg_pool2d(img2, kernel_size=2, stride=2)
            img1 = filtered_im1
            img2 = filtered_im2

        value = (torch.prod(mcs[0:levels-1] ** self.weight[0:levels-1])*
                                    (msssim[levels-1] ** self.weight[levels-1]))
        return value


    def forward(self, img1, img2):
        img1 = torch.relu(img1)
        img2 = torch.relu(img2)
        return self.loss_weight*(1-self.ms_ssim(img1, img2))