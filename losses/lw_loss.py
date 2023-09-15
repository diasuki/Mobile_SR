import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from utils.corner import CornerDetectionEigenVals

class LWLoss(_Loss):
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

            >>> lw = LWLoss()
            >>> input = torch.randn((1,3,32,32), requires_grad=True)
            >>> target = torch.randn((1,3,32,32)
            >>> loss = lw(input, target)
            >>> loss.backward()
        """

    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean', lambda_weight=4) -> None:
        super(LWLoss, self).__init__(size_average, reduce, reduction)

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