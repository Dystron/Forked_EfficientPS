import torch
import torch.nn as nn
import numpy as np
from ..registry import LOSSES
from .utils import weighted_loss


def epsilon(omega, omega_p, delta_p, a1):
    omega_hat = 2 * (delta_p - a1)
    x = (omega - omega_hat) / 2
    return bbox_loss(x, omega, omega_p)


def epsilon_prime(omega, omega_p, delta_p, a1):
    omega_hat = 2 * (delta_p - a1)
    x = (omega - omega_hat) / 2
    diff = np.log(omega) - np.log(omega_p)
    return 0.5 * smooth_l1_loss_prime(, beta) + smooth_l1_loss_prime(diff, beta) / omega



def sigma():
    pass

def phi():
    pass


def bbox_loss(x, omega, omega_p, beta=1.0):
    diff = np.log(omega) - np.log(omega_p)
    return smooth_l1_loss(x, beta) + smooth_l1_loss(diff, beta)


def smooth_l1_loss(x, beta=1.0):
    assert beta > 0
    diff = torch.abs(x)
    loss = torch.where(diff < beta, 0.5 * diff * diff / beta,
                       diff - 0.5 * beta)
    return loss


def smooth_l1_loss_prime(x, beta):
    return np.clip(x / beta, a_min=-1, a_max=1)


@LOSSES.register_module
class cabb(nn.Module):
    """
    loss_bbox:
    cls_score, bbox_pred, *bbox_targets
    """

    def __init__(self, beta=1.0, reduction='mean', loss_weight=1.0):
        super(cabb, self).__init__()
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                img_shapes,
                proposal_list,
                sampling_results,
                cases,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        # TODO debug: test if the crop values or cases are as long as the number of images
        #  (otherwise check valid ids in transforms)
        # print(f'target\n{target}')
        print(f'img shape\n{img_shapes}')
        print(f'proposal_list\n{proposal_list}')
        print(f'sampling_results\n{sampling_results}')
        print(f'len of proposal list and sampling results pos_inds \n{proposal_list[0].shape} {sampling_results[0].pos_inds.shape}')
        # are those our anchors and pixel coordinates?
        positive_proposals = proposal_list[0][sampling_results[0].pos_inds, :]
        print(f'positive proposals\n{positive_proposals}')
        print(f'cases in loss\n'
              f'{cases}')
        # assert reduction_override in (None, 'none', 'mean', 'sum')
        # what is this ?y
        reduction = (
            reduction_override if reduction_override else self.reduction)
        print(f'Prediction size: {pred.size()}')
        print('Target size: {}'.format(str(target.size())))

        loss_bbox = self.loss_weight * smooth_l1_loss(
            pred,
            target,
            weight,
            beta=self.beta,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss_bbox
