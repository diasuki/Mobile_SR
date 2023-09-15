import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
import cv2
import numpy as np
from utils.corner import CornerDetectionEigenVals

class Loss1(_Loss):
    r"""
        Args:
            size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
                the losses are averaged over each loss element in the batch. Note that for
                some losses, there are multiple elements per sample. If the field :attr:`size_average`
                is set to ``False``, the losses are instead summed for each minibatch. Ignored
                when :attr:`reduce` is ``False``. Default: ``True``
            reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
                losses are averaged or summed over observations for each minibatch depending
                on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
                batch element instead and ignores :attr:`size_average`. Default: ``True``
            reduction (string, optional): Specifies the reduction to apply to the output:
                ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
                ``'mean'``: the sum of the output will be divided by the number of
                elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
                and :attr:`reduce` are in the process of being deprecated, and in the meantime,
                specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``

        Shape:
            - Input: :math:`(N, C, H, W)` where :math:`*` means, any number of additional
              dimensions
            - Target: :math:`(N, C, H, W)`, same shape as the input
            - Output: scalar. If :attr:`reduction` is ``'none'``, then
              :math:`(N, C, H, W)`, same shape as the input

        Examples::

            >>> gw = GWLoss()
            >>> input = torch.randn((1,3,32,32), requires_grad=True)
            >>> target = torch.randn((1,3,32,32)
            >>> loss = gw(input, target)
            >>> loss.backward()
        """

    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(Loss1, self).__init__(size_average, reduce, reduction)

        sobel_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        sobel_y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
        self.sobel_x = torch.FloatTensor(sobel_x).cuda()
        self.sobel_y = torch.FloatTensor(sobel_y).cuda()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        b, c, w, h = input.shape

        weight_x = self.sobel_x.expand(c, 1, 3, 3)
        weight_y = self.sobel_y.expand(c, 1, 3, 3)

        Ix1 = F.conv2d(input, weight_x, stride=1, padding=1, groups=c)
        Ix2 = F.conv2d(target, weight_x, stride=1, padding=1, groups=c)
        Iy1 = F.conv2d(input, weight_y, stride=1, padding=1, groups=c)
        Iy2 = F.conv2d(target, weight_y, stride=1, padding=1, groups=c)
        dx = torch.abs(Ix1 - Ix2)
        dy = torch.abs(Iy1 - Iy2)

        lambda1 = torch.zeros_like(input).to(input.device)
        lambda2 = torch.zeros_like(input).to(input.device)
        target_Y = 0.257*target[:, :1, :, :] + 0.564*target[:, 1:2, :, :] + 0.098*target[:, 2:, :, :] + 16/255.0
        target_Y_numpy = np.transpose(target_Y.cpu().numpy(), (0, 2, 3, 1))
        for i in range(b):
            dst = cv2.cornerEigenValsAndVecs(target_Y_numpy[i, :, :, 0], 3, 3)
            lambda1[i, :, :, :] = torch.from_numpy(dst[:,:,0])
            lambda2[i, :, :, :] = torch.from_numpy(dst[:,:,1])

        loss = (1 + 4 * dx) * (1 + 4 * dy) * (1 + lambda1 + lambda2) * torch.abs(input - target)

        if self.reduction=='mean':
            return torch.mean(loss)
        elif self.reduction=='sum':
            return torch.sum(loss)
        else:
            return loss


class Loss2(_Loss):
    r"""
        Args:
            size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
                the losses are averaged over each loss element in the batch. Note that for
                some losses, there are multiple elements per sample. If the field :attr:`size_average`
                is set to ``False``, the losses are instead summed for each minibatch. Ignored
                when :attr:`reduce` is ``False``. Default: ``True``
            reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
                losses are averaged or summed over observations for each minibatch depending
                on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
                batch element instead and ignores :attr:`size_average`. Default: ``True``
            reduction (string, optional): Specifies the reduction to apply to the output:
                ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
                ``'mean'``: the sum of the output will be divided by the number of
                elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
                and :attr:`reduce` are in the process of being deprecated, and in the meantime,
                specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``

        Shape:
            - Input: :math:`(N, C, H, W)` where :math:`*` means, any number of additional
              dimensions
            - Target: :math:`(N, C, H, W)`, same shape as the input
            - Output: scalar. If :attr:`reduction` is ``'none'``, then
              :math:`(N, C, H, W)`, same shape as the input

        Examples::

            >>> gw = GWLoss()
            >>> input = torch.randn((1,3,32,32), requires_grad=True)
            >>> target = torch.randn((1,3,32,32)
            >>> loss = gw(input, target)
            >>> loss.backward()
        """

    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(Loss2, self).__init__(size_average, reduce, reduction)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        b, c, w, h = input.shape

        lambda1 = torch.zeros_like(input).to(input.device)
        lambda2 = torch.zeros_like(input).to(input.device)
        target_Y = 0.257*target[:, :1, :, :] + 0.564*target[:, 1:2, :, :] + 0.098*target[:, 2:, :, :] + 16/255.0
        target_Y_numpy = np.transpose(target_Y.cpu().numpy(), (0, 2, 3, 1))
        for i in range(b):
            dst = cv2.cornerEigenValsAndVecs(target_Y_numpy[i, :, :, 0], 3, 3)
            lambda1[i, :, :, :] = torch.from_numpy(dst[:,:,0])
            lambda2[i, :, :, :] = torch.from_numpy(dst[:,:,1])

        loss = (1 + lambda1 + lambda2) * torch.abs(input - target)

        if self.reduction=='mean':
            return torch.mean(loss)
        elif self.reduction=='sum':
            return torch.sum(loss)
        else:
            return loss


class Loss3(_Loss):
    r"""
        Args:
            size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
                the losses are averaged over each loss element in the batch. Note that for
                some losses, there are multiple elements per sample. If the field :attr:`size_average`
                is set to ``False``, the losses are instead summed for each minibatch. Ignored
                when :attr:`reduce` is ``False``. Default: ``True``
            reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
                losses are averaged or summed over observations for each minibatch depending
                on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
                batch element instead and ignores :attr:`size_average`. Default: ``True``
            reduction (string, optional): Specifies the reduction to apply to the output:
                ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
                ``'mean'``: the sum of the output will be divided by the number of
                elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
                and :attr:`reduce` are in the process of being deprecated, and in the meantime,
                specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``

        Shape:
            - Input: :math:`(N, C, H, W)` where :math:`*` means, any number of additional
              dimensions
            - Target: :math:`(N, C, H, W)`, same shape as the input
            - Output: scalar. If :attr:`reduction` is ``'none'``, then
              :math:`(N, C, H, W)`, same shape as the input

        Examples::

            >>> gw = GWLoss()
            >>> input = torch.randn((1,3,32,32), requires_grad=True)
            >>> target = torch.randn((1,3,32,32)
            >>> loss = gw(input, target)
            >>> loss.backward()
        """

    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(Loss3, self).__init__(size_average, reduce, reduction)

        sobel_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        sobel_y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
        self.sobel_x = torch.FloatTensor(sobel_x).cuda()
        self.sobel_y = torch.FloatTensor(sobel_y).cuda()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        b, c, w, h = input.shape

        weight_x = self.sobel_x.expand(c, 1, 3, 3)
        weight_y = self.sobel_y.expand(c, 1, 3, 3)

        Ix1 = F.conv2d(input, weight_x, stride=1, padding=1, groups=c)
        Ix2 = F.conv2d(target, weight_x, stride=1, padding=1, groups=c)
        Iy1 = F.conv2d(input, weight_y, stride=1, padding=1, groups=c)
        Iy2 = F.conv2d(target, weight_y, stride=1, padding=1, groups=c)
        dx = torch.abs(Ix1 - Ix2)
        dy = torch.abs(Iy1 - Iy2)

        lambda1 = torch.zeros_like(input).to(input.device)
        lambda2 = torch.zeros_like(input).to(input.device)
        target_Y = 0.257*target[:, :1, :, :] + 0.564*target[:, 1:2, :, :] + 0.098*target[:, 2:, :, :] + 16/255.0
        target_Y_numpy = np.transpose(target_Y.cpu().numpy(), (0, 2, 3, 1))
        for i in range(b):
            dst = cv2.cornerEigenValsAndVecs(target_Y_numpy[i, :, :, 0], 3, 3)
            lambda1[i, :, :, :] = torch.from_numpy(dst[:,:,0])
            lambda2[i, :, :, :] = torch.from_numpy(dst[:,:,1])

        loss = ((1 + dx ** (1 + lambda1)) * (1 + dy ** (1 + lambda2)) + (1 + dx ** (1 + lambda2)) * (1 + dy ** (1 + lambda1))) / 2 * torch.abs(input - target)

        if self.reduction=='mean':
            return torch.mean(loss)
        elif self.reduction=='sum':
            return torch.sum(loss)
        else:
            return loss


class Loss4(_Loss):
    r"""
        Args:
            size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
                the losses are averaged over each loss element in the batch. Note that for
                some losses, there are multiple elements per sample. If the field :attr:`size_average`
                is set to ``False``, the losses are instead summed for each minibatch. Ignored
                when :attr:`reduce` is ``False``. Default: ``True``
            reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
                losses are averaged or summed over observations for each minibatch depending
                on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
                batch element instead and ignores :attr:`size_average`. Default: ``True``
            reduction (string, optional): Specifies the reduction to apply to the output:
                ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
                ``'mean'``: the sum of the output will be divided by the number of
                elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
                and :attr:`reduce` are in the process of being deprecated, and in the meantime,
                specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``

        Shape:
            - Input: :math:`(N, C, H, W)` where :math:`*` means, any number of additional
              dimensions
            - Target: :math:`(N, C, H, W)`, same shape as the input
            - Output: scalar. If :attr:`reduction` is ``'none'``, then
              :math:`(N, C, H, W)`, same shape as the input

        Examples::

            >>> gw = GWLoss()
            >>> input = torch.randn((1,3,32,32), requires_grad=True)
            >>> target = torch.randn((1,3,32,32)
            >>> loss = gw(input, target)
            >>> loss.backward()
        """

    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean', lambda_weight=4) -> None:
        super(Loss4, self).__init__(size_average, reduce, reduction)

        self.lambda_weight = lambda_weight

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        b, _, _, _ = input.shape

        input_numpy = np.transpose(input.cpu().detach().numpy() * 255., (0, 2, 3, 1)).astype(np.uint8)
        target_numpy = np.transpose(target.cpu().numpy() * 255., (0, 2, 3, 1)).astype(np.uint8)
        
        input_lambda1 = torch.zeros_like(input).to(input.device)
        input_lambda2 = torch.zeros_like(input).to(input.device)
        target_lambda1 = torch.zeros_like(input).to(input.device)
        target_lambda2 = torch.zeros_like(input).to(input.device)
        for i in range(b):
            input_dst = cv2.cornerEigenValsAndVecs(np.float32(cv2.cvtColor(input_numpy[i, :, :, :], cv2.COLOR_BGR2GRAY)), 3, 3)
            input_lambda1[i, :, :, :] = torch.from_numpy(cv2.normalize(input_dst[:,:,0], None, 0, 1, cv2.NORM_MINMAX))
            input_lambda2[i, :, :, :] = torch.from_numpy(cv2.normalize(input_dst[:,:,1], None, 0, 1, cv2.NORM_MINMAX))
            target_dst = cv2.cornerEigenValsAndVecs(np.float32(cv2.cvtColor(target_numpy[i, :, :, :], cv2.COLOR_BGR2GRAY)), 3, 3)
            target_lambda1[i, :, :, :] = torch.from_numpy(cv2.normalize(target_dst[:,:,0], None, 0, 1, cv2.NORM_MINMAX))
            target_lambda2[i, :, :, :] = torch.from_numpy(cv2.normalize(target_dst[:,:,1], None, 0, 1, cv2.NORM_MINMAX))

        # print(input_lambda1.max().item(), input_lambda1.min().item(), input_lambda2.max().item(), input_lambda2.min().item())

        dx = torch.abs(input_lambda1 - target_lambda1)
        dy = torch.abs(input_lambda2 - target_lambda2)

        loss = (1 + self.lambda_weight * dx) * (1 + self.lambda_weight * dy) * torch.abs(input - target)

        if self.reduction=='mean':
            return torch.mean(loss)
        elif self.reduction=='sum':
            return torch.sum(loss)
        else:
            return loss


class Loss5(_Loss):
    r"""
        Args:
            size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
                the losses are averaged over each loss element in the batch. Note that for
                some losses, there are multiple elements per sample. If the field :attr:`size_average`
                is set to ``False``, the losses are instead summed for each minibatch. Ignored
                when :attr:`reduce` is ``False``. Default: ``True``
            reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
                losses are averaged or summed over observations for each minibatch depending
                on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
                batch element instead and ignores :attr:`size_average`. Default: ``True``
            reduction (string, optional): Specifies the reduction to apply to the output:
                ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
                ``'mean'``: the sum of the output will be divided by the number of
                elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
                and :attr:`reduce` are in the process of being deprecated, and in the meantime,
                specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``

        Shape:
            - Input: :math:`(N, C, H, W)` where :math:`*` means, any number of additional
              dimensions
            - Target: :math:`(N, C, H, W)`, same shape as the input
            - Output: scalar. If :attr:`reduction` is ``'none'``, then
              :math:`(N, C, H, W)`, same shape as the input

        Examples::

            >>> gw = GWLoss()
            >>> input = torch.randn((1,3,32,32), requires_grad=True)
            >>> target = torch.randn((1,3,32,32)
            >>> loss = gw(input, target)
            >>> loss.backward()
        """

    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean', lambda_weight=4) -> None:
        super(Loss5, self).__init__(size_average, reduce, reduction)

        self.lambda_weight = lambda_weight
        self.eigen_vals = CornerDetectionEigenVals()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        input_lambda1, input_lambda2 = self.eigen_vals(input)
        target_lambda1, target_lambda2 = self.eigen_vals(target)
        
        dx = torch.abs(input_lambda1 - target_lambda1)
        dy = torch.abs(input_lambda2 - target_lambda2)

        loss = (1 + self.lambda_weight * dx) * (1 + self.lambda_weight * dy) * torch.abs(input - target)

        if self.reduction=='mean':
            return torch.mean(loss)
        elif self.reduction=='sum':
            return torch.sum(loss)
        else:
            return loss


class Loss6(_Loss):
    r"""
        Args:
            size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
                the losses are averaged over each loss element in the batch. Note that for
                some losses, there are multiple elements per sample. If the field :attr:`size_average`
                is set to ``False``, the losses are instead summed for each minibatch. Ignored
                when :attr:`reduce` is ``False``. Default: ``True``
            reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
                losses are averaged or summed over observations for each minibatch depending
                on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
                batch element instead and ignores :attr:`size_average`. Default: ``True``
            reduction (string, optional): Specifies the reduction to apply to the output:
                ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
                ``'mean'``: the sum of the output will be divided by the number of
                elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
                and :attr:`reduce` are in the process of being deprecated, and in the meantime,
                specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``

        Shape:
            - Input: :math:`(N, C, H, W)` where :math:`*` means, any number of additional
              dimensions
            - Target: :math:`(N, C, H, W)`, same shape as the input
            - Output: scalar. If :attr:`reduction` is ``'none'``, then
              :math:`(N, C, H, W)`, same shape as the input

        Examples::

            >>> gw = GWLoss()
            >>> input = torch.randn((1,3,32,32), requires_grad=True)
            >>> target = torch.randn((1,3,32,32)
            >>> loss = gw(input, target)
            >>> loss.backward()
        """

    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(Loss6, self).__init__(size_average, reduce, reduction)

        sobel_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        sobel_y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
        self.sobel_x = torch.FloatTensor(sobel_x)
        self.sobel_y = torch.FloatTensor(sobel_y)
        self.eigen_vals = CornerDetectionEigenVals()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        b, c, w, h = input.shape

        weight_x = self.sobel_x.expand(c, 1, 3, 3).to(input.device)
        weight_y = self.sobel_y.expand(c, 1, 3, 3).to(input.device)

        Ix1 = F.conv2d(input, weight_x, stride=1, padding=1, groups=c)
        Ix2 = F.conv2d(target, weight_x, stride=1, padding=1, groups=c)
        Iy1 = F.conv2d(input, weight_y, stride=1, padding=1, groups=c)
        Iy2 = F.conv2d(target, weight_y, stride=1, padding=1, groups=c)
        dx = torch.abs(Ix1 - Ix2)
        dy = torch.abs(Iy1 - Iy2)

        input_lambda1, input_lambda2 = self.eigen_vals(input)
        target_lambda1, target_lambda2 = self.eigen_vals(target)
        
        dx1 = torch.abs(input_lambda1 - target_lambda1)
        dy1 = torch.abs(input_lambda2 - target_lambda2)

        loss = (1 + 4 * dx + 4 * dx1) * (1 + 4 * dy + 4 * dy1) * torch.abs(input - target)

        if self.reduction=='mean':
            return torch.mean(loss)
        elif self.reduction=='sum':
            return torch.sum(loss)
        else:
            return loss


class Loss7(_Loss):
    r"""
        Args:
            size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
                the losses are averaged over each loss element in the batch. Note that for
                some losses, there are multiple elements per sample. If the field :attr:`size_average`
                is set to ``False``, the losses are instead summed for each minibatch. Ignored
                when :attr:`reduce` is ``False``. Default: ``True``
            reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
                losses are averaged or summed over observations for each minibatch depending
                on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
                batch element instead and ignores :attr:`size_average`. Default: ``True``
            reduction (string, optional): Specifies the reduction to apply to the output:
                ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
                ``'mean'``: the sum of the output will be divided by the number of
                elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
                and :attr:`reduce` are in the process of being deprecated, and in the meantime,
                specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``

        Shape:
            - Input: :math:`(N, C, H, W)` where :math:`*` means, any number of additional
              dimensions
            - Target: :math:`(N, C, H, W)`, same shape as the input
            - Output: scalar. If :attr:`reduction` is ``'none'``, then
              :math:`(N, C, H, W)`, same shape as the input

        Examples::

            >>> gw = GWLoss()
            >>> input = torch.randn((1,3,32,32), requires_grad=True)
            >>> target = torch.randn((1,3,32,32)
            >>> loss = gw(input, target)
            >>> loss.backward()
        """

    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean', lambda_weight=4) -> None:
        super(Loss7, self).__init__(size_average, reduce, reduction)

        sobel_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        sobel_y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
        self.sobel_x = torch.FloatTensor(sobel_x)
        self.sobel_y = torch.FloatTensor(sobel_y)
        self.eigen_vals = CornerDetectionEigenVals()
        self.lambda_weight = lambda_weight

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        b, c, w, h = input.shape

        weight_x = self.sobel_x.expand(c, 1, 3, 3).to(input.device)
        weight_y = self.sobel_y.expand(c, 1, 3, 3).to(input.device)

        Ix1 = F.conv2d(input, weight_x, stride=1, padding=1, groups=c)
        Ix2 = F.conv2d(target, weight_x, stride=1, padding=1, groups=c)
        Iy1 = F.conv2d(input, weight_y, stride=1, padding=1, groups=c)
        Iy2 = F.conv2d(target, weight_y, stride=1, padding=1, groups=c)
        dx = torch.abs(Ix1 - Ix2)
        dy = torch.abs(Iy1 - Iy2)

        input_lambda1, input_lambda2 = self.eigen_vals(input)
        target_lambda1, target_lambda2 = self.eigen_vals(target)
        
        dx1 = torch.abs(input_lambda1 - target_lambda1)
        dy1 = torch.abs(input_lambda2 - target_lambda2)

        loss = (1 + 4 * dx) * (1 + 4 * dy) * (1 + self.lambda_weight * dx1) * (1 + self.lambda_weight * dy1) * torch.abs(input - target)

        if self.reduction=='mean':
            return torch.mean(loss)
        elif self.reduction=='sum':
            return torch.sum(loss)
        else:
            return loss


class Loss8(_Loss):
    r"""
        Args:
            size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
                the losses are averaged over each loss element in the batch. Note that for
                some losses, there are multiple elements per sample. If the field :attr:`size_average`
                is set to ``False``, the losses are instead summed for each minibatch. Ignored
                when :attr:`reduce` is ``False``. Default: ``True``
            reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
                losses are averaged or summed over observations for each minibatch depending
                on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
                batch element instead and ignores :attr:`size_average`. Default: ``True``
            reduction (string, optional): Specifies the reduction to apply to the output:
                ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
                ``'mean'``: the sum of the output will be divided by the number of
                elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
                and :attr:`reduce` are in the process of being deprecated, and in the meantime,
                specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``

        Shape:
            - Input: :math:`(N, C, H, W)` where :math:`*` means, any number of additional
              dimensions
            - Target: :math:`(N, C, H, W)`, same shape as the input
            - Output: scalar. If :attr:`reduction` is ``'none'``, then
              :math:`(N, C, H, W)`, same shape as the input

        Examples::

            >>> gw = GWLoss()
            >>> input = torch.randn((1,3,32,32), requires_grad=True)
            >>> target = torch.randn((1,3,32,32)
            >>> loss = gw(input, target)
            >>> loss.backward()
        """

    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean', lambda_weight=4) -> None:
        super(Loss8, self).__init__(size_average, reduce, reduction)

        self.lambda_weight = lambda_weight
        sobel_x = torch.FloatTensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = torch.FloatTensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        self.eigen_vals = CornerDetectionEigenVals(gradient_tensor=torch.stack([sobel_x, sobel_y]).unsqueeze(1))

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        input_lambda1, input_lambda2 = self.eigen_vals(input)
        target_lambda1, target_lambda2 = self.eigen_vals(target)
        
        dx = torch.abs(input_lambda1 - target_lambda1)
        dy = torch.abs(input_lambda2 - target_lambda2)

        loss = (1 + self.lambda_weight * dx) * (1 + self.lambda_weight * dy) * torch.abs(input - target)

        if self.reduction=='mean':
            return torch.mean(loss)
        elif self.reduction=='sum':
            return torch.sum(loss)
        else:
            return loss