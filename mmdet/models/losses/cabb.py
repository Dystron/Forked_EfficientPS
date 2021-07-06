import torch
import torch.nn as nn
import numpy as np
from ..registry import LOSSES
from .utils import weighted_loss


def epsilon(omega, omega_p, omega_hat, beta=1.0):
    x = (omega - omega_hat) / 2
    return bbox_loss(x, omega, omega_p)


def epsilon_prime(omega, omega_p, omega_hat, beta=1.0):
    x = (omega - omega_hat) / 2
    diff = np.log(omega) - np.log(omega_p)
    return 0.5 * smooth_l1_loss_prime(x, beta) + smooth_l1_loss_prime(diff, beta) / omega


# phi is objective function of O2 which we never use
# def phi(omega, omega_p, delta, delta_p, a1, beta=1.0):
#     return smooth_l1_loss(delta - delta_p, beta) + smooth_l1_loss(np.log(omega) - np.log(omega_p))


def sigma(omega, omega_p, omega_hat, beta=1.0):
    return omega * epsilon_prime(omega, omega_p, omega_hat, beta)


def bbox_loss(x, omega, omega_p, beta=1.0):
    diff = np.log(omega) - np.log(omega_p)
    return smooth_l1_loss(x, beta) + smooth_l1_loss(diff, beta)


def smooth_l1_loss(x, beta=1.0):
    assert beta > 0
    diff = torch.abs(x)
    loss = torch.where(diff < beta, 0.5 * diff * diff / beta,
                       diff - 0.5 * beta)
    return loss


def smooth_l1_loss_prime(x, beta=1.0):
    return np.clip(x / beta, a_min=-1, a_max=1)


def FIND_MIN(interval, objective_func, omega_p, omega_hat, beta=1.0, eps_threshold=10e-30):
    # interval is min and max for omega
    u, v = interval
    if objective_func(omega=u, omega_p=omega_p, omega_hat=omega_hat,
                      beta=beta) >= 0:
        return u
    elif objective_func(v, omega_p=omega_p, omega_hat=omega_hat,
                        beta=beta) <= 0:
        return v
    else:
        m = u + v / 2
        if v - u < eps_threshold:
            return m
        elif objective_func(m) >= 0:
            return FIND_MIN([u, m], objective_func, omega_p, omega_hat, beta)
        else:
            return FIND_MIN([m, v], objective_func, omega_p, omega_hat, beta)


def J_getter(i, omega_0=None, omega_p=None, beta=None, omega_hat=None):
    if i == 1:
        return [max(omega_0, omega_p), min(np.e ** min(beta, 1) * omega_p, omega_hat)]
    if i == 2:
        return [max(omega_0, 2 * np.sqrt(beta), omega_hat - 2 * beta, np.e ** beta * omega_p), omega_hat]
    if i == 3:
        return [max(omega_0, omega_hat - 2 * beta, np.e * omega_p), min(np.e ** beta, omega_hat)]
    if i == 4:
        return [max(omega_0, omega_hat - 2 * beta, np.e * omega_p),
                min(np.e ** beta * omega_p, omega_hat, (omega_hat / 4) * (1 + np.sqrt(1 - (32 / omega_hat ** 2))))]
    if i == 5:
        return [max(omega_0, omega_hat - 2 * beta, np.e * omega_p,
                    (omega_hat / 4) * (1 - np.sqrt(1 - (32 / omega_hat ** 2)))), min(np.e ** beta * omega_p, omega_hat)]


def SOLVE_O1(omega_p, omega_hat, a1, b1, beta=1.0):
    omega_0 = b1 - a1
    S = [omega_0]
    if max(omega_0, omega_hat) < omega_p:
        return FIND_MIN(interval=[max(omega_0, omega_hat), omega_p], objective_func=epsilon_prime, omega_p=omega_p,
                        omega_hat=omega_hat, beta=beta)
    elif max(omega_0, omega_p) < omega_hat:
        S += omega_hat, FIND_MIN(J_getter(1, omega_0=omega_0, omega_p=omega_p, beta=beta, omega_hat=omega_hat),
                                 objective_func=epsilon_prime, omega_p=omega_p, omega_hat=omega_hat, beta=beta), \
             FIND_MIN(J_getter(2, omega_0=omega_0, omega_p=omega_p, beta=beta, omega_hat=omega_hat),
                      objective_func=epsilon_prime, omega_p=omega_p, omega_hat=omega_hat, beta=beta)
        if omega_hat <= 4 * np.sqrt(2):
            S += FIND_MIN(J_getter(3, omega_0=omega_0, omega_p=omega_p, beta=beta, omega_hat=omega_hat),
                          objective_func=sigma, omega_p=omega_p, omega_hat=omega_hat, beta=beta)
        else:
            S += FIND_MIN(J_getter(4, omega_0=omega_0, omega_p=omega_p, beta=beta, omega_hat=omega_hat),
                          objective_func=sigma, omega_p=omega_p, omega_hat=omega_hat, beta=beta), FIND_MIN(
                J_getter(5, omega_0=omega_0, omega_p=omega_p, beta=beta, omega_hat=omega_hat),
                objective_func=sigma, omega_p=omega_p, omega_hat=omega_hat, beta=beta)
    return S[np.argmin([epsilon(omega=cur, omega_p=omega_p, omega_hat=omega_hat, beta=beta) for cur in S])]


def SOLVE_O2(delta_p, omega_p, a2, b2, beta=1.0):
    omega_hat1 = 2 * (delta_p - a2)
    omega_hat2 = 2 * (b2 - delta_p)
    if omega_p >= max(omega_hat1, omega_hat2):
        return (delta_p, omega_p)
    omega_1 = SOLVE_O1(omega_p=omega_p, omega_hat=omega_hat1, a1=a2, b1=b2, beta=beta)
    omega_2 = SOLVE_O1(omega_p=omega_p, omega_hat=omega_hat2, a1=a2, b1=b2, beta=beta)
    # we can use omegahat1 here as it has the a2 which is given to eps as a1 and
    # eps is objective function of o1
    # TODO verify what exactly omega_hat should be in the next two calls
    # maybe compute a1 as on page 5 and get the omega_hat that way? (using delta_gt)
    if epsilon(omega=omega_1, omega_p=omega_p, omega_hat=omega_hat1, beta=beta) <= \
            epsilon(omega=omega_2, omega_p=omega_p, omega_hat=omega_hat2, beta=beta):
        return (a2 + (omega_1 / 2), omega_1)
    else:
        return (b2 + (omega_2 / 2), omega_2)


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
                crop_shapes,
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
        print(f'img shape\n{crop_shapes}')
        print(f'proposal_list\n{proposal_list.shape}')
        print(f'sampling_results\n{sampling_results}')
        print(f'cases in loss\n'
              f'{cases.shape}')
        # assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        print(f'Prediction size: {pred.size()}')
        print('Target size: {}'.format(str(target.size())))

        """
        cases:
        [lower x cropped?][upper x cropped?][lower y cropped?][upper y cropped]
        delta notation:
        dx, dy, dw, dh
        """
        # TODO: why are some ground truths = [0, 0, 0, 0] ?
        label = [None, None, None, None]
        for i in range(pred.shape[0]):
            print(f'Bbox prediction: {pred.cpu()[i]}')
            print(f'Associated Proposal: {proposal_list[i]}')
            print(f'Case for this prediction: {cases.cpu()[i]}')
            print(f'GT for this prediction: {target.cpu()[i]}')
            print(f'Crop dimensions: {crop_shapes[i]}')
            # set the x dimension parameters of the target label
            if not cases[0] and not cases[1]:
                # x dimension has not been cut
                label[0] = target[i][0]
                label[2] = target[i][2]
            elif cases[0] and not cases[1]:
                # x cut left --> O1 first case
                # todo: check if this is not second case
                ca = 0.5 * (proposal_list[i][0] + proposal_list[i][2])
                da = proposal_list[i][2] - proposal_list[i][0]
                a1 = target[i][0] - 0.5 * target[i][2]
                b1 = (crop_shapes[i][0] - ca) / da
                omega_hat = 2 * (pred[i][0] - a1)
                omega_star = SOLVE_O1(omega_p=pred[i][2], omega_hat=omega_hat, a1=a1, b1=b1)
                delta_star = a1 + 0.5 * omega_star
                label[0] = delta_star
                label[2] = omega_star
            elif not cases[0] and cases[1]:
                # x cut right --> O1 second case
                # todo: check if this is not first case
                ca = 0.5 * (proposal_list[i][0] + proposal_list[i][2])
                da = proposal_list[i][2] - proposal_list[i][0]
                a1 = - ca / da
                b1 = target[i][0] + 0.5 * target[i][2]
                omega_hat = 2 * (b1 - pred[i][0])
                omega_star = SOLVE_O1(omega_p=pred[i][2], omega_hat=omega_hat, a1=a1, b1=b1)
                delta_star = b1 - 0.5 * omega_star
                label[0] = delta_star
                label[2] = omega_star
            else:
                # x cut on both sides --> O2
                ca = 0.5 * (proposal_list[i][0] + proposal_list[i][2])
                da = proposal_list[i][2] - proposal_list[i][0]
                a2 = - ca / da
                b2 = (crop_shapes[i][0] - ca) / da
                label[0], label[2] = SOLVE_O2(delta_p=pred[i][0], omega_p=pred[i][0], a2=a2, b2=b2)

            # set the y dimension parameters of the target label
            if not cases[2] and not cases[3]:
                # x dimension has not been cut
                label[1] = target[i][1]
                label[3] = target[i][3]
            elif cases[2] and not cases[3]:
                # x cut left
                pass
            elif not cases[2] and cases[3]:
                # x cut right
                pass
            else:
                # x cut on both sides
                pass

            # MORE EFFICIENT SLICE IN X DIM -->4 CASES THEN CONCAT THEN SLICE IN Y DIM --> 4 CASES


        """loss_bbox = self.loss_weight * smooth_l1_loss(
            pred,
            target,
            weight,
            beta=self.beta,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss_bbox"""
        return 1