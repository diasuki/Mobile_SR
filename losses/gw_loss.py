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