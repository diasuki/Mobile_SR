import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

class GWLoss(_Loss):
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
        super(GWLoss, self).__init__(size_average, reduce, reduction)

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
        loss = (1 + 4 * dx) * (1 + 4 * dy) * torch.abs(input - target)

        if self.reduction=='mean':
            return torch.mean(loss)
        elif self.reduction=='sum':
            return torch.sum(loss)
        else:
            return loss


class Adaptive_GWLoss(_Loss):
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
        super(Adaptive_GWLoss, self).__init__(size_average, reduce, reduction)

        sobel_0 = [[-1,0,1],[-2,0,2],[-1,0,1]]
        sobel_90 = [[-1,-2,-1],[0,0,0],[1,2,1]]
        sobel_45 = [[-2,-1,0],[-1,0,1],[0,1,2]]
        sobel_135 = [[0,-1,-2],[1,0,-1],[2,1,0]]
        self.sobel_0 = torch.FloatTensor(sobel_0).cuda()
        self.sobel_90 = torch.FloatTensor(sobel_90).cuda()
        self.sobel_45 = torch.FloatTensor(sobel_45).cuda()
        self.sobel_135 = torch.FloatTensor(sobel_135).cuda()

    def forward(self, input: torch.Tensor, target: torch.Tensor, corner=True) -> torch.Tensor:
        Y_x1 = torch.mean(input, dim=1, keepdim=True)
        Y_x2 = torch.mean(target, dim=1, keepdim=True)

        b, c, w, h = Y_x1.shape

        sobel_0 = self.sobel_0.expand(c, 1, 3, 3)
        sobel_90 = self.sobel_90.expand(c, 1, 3, 3)
        sobel_45 = self.sobel_45.expand(c, 1, 3, 3)
        sobel_135 = self.sobel_135.expand(c, 1, 3, 3)
        sobel_0 = sobel_0.type_as(Y_x1)
        sobel_90 = sobel_90.type_as(Y_x1)
        sobel_45 = sobel_0.type_as(Y_x1)
        sobel_135 = sobel_90.type_as(Y_x1)

        weight_0 = nn.Parameter(data=sobel_0, requires_grad=False)
        weight_90 = nn.Parameter(data=sobel_90, requires_grad=False)
        weight_45 = nn.Parameter(data=sobel_45, requires_grad=False)
        weight_135 = nn.Parameter(data=sobel_135, requires_grad=False)

        I1_0 = F.conv2d(Y_x1, weight_0, stride=1, padding=1, groups=c)
        I2_0 = F.conv2d(Y_x2, weight_0, stride=1, padding=1, groups=c)
        I1_90 = F.conv2d(Y_x1, weight_90, stride=1, padding=1, groups=c)
        I2_90 = F.conv2d(Y_x2, weight_90, stride=1, padding=1, groups=c)
        I1_45 = F.conv2d(Y_x1, weight_45, stride=1, padding=1, groups=c)
        I2_45 = F.conv2d(Y_x2, weight_45, stride=1, padding=1, groups=c)
        I1_135 = F.conv2d(Y_x1, weight_135, stride=1, padding=1, groups=c)
        I2_135 = F.conv2d(Y_x2, weight_135, stride=1, padding=1, groups=c)
        d0 = torch.abs(I1_0 - I2_0)
        d90 = torch.abs(I1_90 - I2_90)
        d45 = torch.abs(I1_45 - I2_45)
        d135 = torch.abs(I1_135 - I2_135)

        if corner:
            d0 = d0.expand(input.shape)
            d90 = d90.expand(input.shape)
            d45 = d45.expand(input.shape)
            d135 = d135.expand(input.shape)
            loss = (1 + 4*d0) * (1 + 4*d90) *(1 + 4*d45) *(1 + 4*d135) * torch.abs(input - target)
        else:
            d = torch.cat((d0, d90, d45, d135), dim=1)
            d = torch.max(d, dim=1, keepdim=True)[0]
            d = d.expand(input.shape)
            loss = (1 + 4*d) * torch.abs(input - target)

        if self.reduction=='mean':
            return torch.mean(loss)
        elif self.reduction=='sum':
            return torch.sum(loss)
        else:
            return loss


class LapGWLoss(_Loss):
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
        super(LapGWLoss, self).__init__(size_average, reduce, reduction)

        sobel = [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]
        self.sobel = torch.FloatTensor(sobel).cuda()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        b, c, w, h = input.shape

        weight = self.sobel.expand(c, 1, 3, 3)

        Ix1 = F.conv2d(input, weight, stride=1, padding=1, groups=c)
        Ix2 = F.conv2d(target, weight, stride=1, padding=1, groups=c)
        dx = torch.abs(Ix1 - Ix2)
        loss = (1 + 4 * dx) * torch.abs(input - target)

        if self.reduction=='mean':
            return torch.mean(loss)
        elif self.reduction=='sum':
            return torch.sum(loss)
        else:
            return loss