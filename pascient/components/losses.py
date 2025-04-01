import torch.nn as nn
import torch

class CrossEntropyLossViews(nn.CrossEntropyLoss):
    """
    CrossEntropyLoss that takes into account the multiple views that are predicted by the model.
    """
    def __init__(self, weight = None, **kwargs):
        if weight is not None:
            weight = 1/torch.Tensor(weight)
        super(CrossEntropyLossViews, self).__init__(weight=weight, **kwargs)
    def forward(self, input, target):
        """
        Computes the cross-entropy loss over the multiple views.

        Input:
        - input [N, V, D] where V is the number of views, D is the number of classes
        - target [N, 1]
        """
        assert target.shape[1] == 1
        target = target.squeeze(1).long()
        losses_per_view = [super(CrossEntropyLossViews,self).forward(input[:, i, :], target).mean() for i in range(input.size(1))]
        return sum(losses_per_view) / len(losses_per_view)


class MSELossViews(nn.MSELoss):
    """
    MSELoss that takes into account the multiple views that are predicted by the model.
    """
    def __init__(self, **kwargs):
        super(MSELossViews, self).__init__( **kwargs)

    def forward(self, input, target):
        """
        Computes the cross-entropy loss over the multiple views.

        Input:
        - input [N, V, D] where V is the number of views, D is the number of classes
        - target [N, D]
        """
        target = target.unsqueeze(1)
        losses_per_view = [super(MSELossViews,self).forward(input[:, i, :], target) for i in range(input.size(1))]
        return sum(losses_per_view) / len(losses_per_view)