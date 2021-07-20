import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from mmdet.core import (delta2bbox, bbox2delta)
from .utils import weighted_loss
from ..registry import LOSSES


def xi(omega, omega_p, omega_hat):
    x = (omega - omega_hat) / 2
    return bbox_loss(x, omega, omega_p)


def xi_prime(omega, omega_p, omega_hat, beta=1.0):
    x = (omega - omega_hat) / 2
    diff = torch.log(omega) - torch.log(omega_p)
    return 0.5 * smooth_l1_loss_prime(x, beta) + smooth_l1_loss_prime(diff, beta) / omega


def sigma(omega, omega_p, omega_hat, beta=1.0):
    return omega * xi_prime(omega, omega_p, omega_hat, beta)


def bbox_loss(x, omega, omega_p, beta=1.0):
    diff = torch.log(omega_p) - torch.log(omega)
    return smooth_l1_loss(x, beta) + smooth_l1_loss(diff, beta)


@weighted_loss
def batched_bbox_loss(pred, target, proposal_list, cases, crop_shapes, crop_info, plot, beta):
    # returns a batch sized loss collection
    losses = torch.zeros([pred.shape[0], 4], dtype=torch.float32, device="cuda")
    for i in range(pred.shape[0]):
        label = [None, None, None, None]
        # optimize in x
        label[0], label[2] = case_distinction(pred[i], proposal_list[i], cases[i], target[i],
                                              crop_shapes[i], axis=0, beta=beta)
        # optimize in y
        label[1], label[3] = case_distinction(pred[i], proposal_list[i], cases[i], target[i],
                                              crop_shapes[i], axis=1, beta=beta)
        label = torch.tensor(label, device="cuda")
        # we dont need to do log of label even tough it is not in log notation
        # as there is a log inside the loss function
        losses[i] = smooth_l1_loss_original(pred[i], target[i], beta=beta)
        if plot:
            # we log the label only for the plotting og the loss
            label[[2, 3]] = torch.log(label[[2, 3]])
            pred[i][[2, 3]] = torch.log(pred[i][[2, 3]])
            target[i][[2, 3]] = torch.log(target[i][[2, 3]])

            plot_anchors_and_gt(crop_info["orig_image"], target[i], proposal_list[i], label, pred[i],
                                *crop_info["crop_left_top"], crop_shapes[i], crop_info["cases"][i])
    return losses


def smooth_l1_loss_original(pred, target, beta=1.0):
    assert beta > 0
    assert pred.size() == target.size() and target.numel() > 0
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * diff * diff / beta,
                       diff - 0.5 * beta)
    return loss


def smooth_l1_loss(x, beta=1.0):
    assert beta > 0
    diff = torch.abs(x)
    loss = torch.where(diff < beta, (diff * diff) / (beta * 2), diff - 0.5 * beta)
    return loss


def smooth_l1_loss_prime(x, beta=1.0):
    return torch.clip(x / beta, min=-1, max=1)


def FIND_MIN(interval, objective_func, omega_p, omega_hat, beta=1.0, eps_threshold=0.1):
    # interval is min and max for omega
    u, v = interval
    if objective_func(omega=u, omega_p=omega_p, omega_hat=omega_hat, beta=beta) >= 0:
        return u
    elif objective_func(v, omega_p=omega_p, omega_hat=omega_hat, beta=beta) <= 0:
        return v
    else:
        m = (u + v) / 2
        if v - u < eps_threshold:
            return m
        elif objective_func(m, omega_p=omega_p, omega_hat=omega_hat, beta=beta) >= 0:
            return FIND_MIN([u, m], objective_func, omega_p, omega_hat, beta)
        else:
            return FIND_MIN([m, v], objective_func, omega_p, omega_hat, beta)


def J_getter(i, omega_0=None, omega_p=None, beta=None, omega_hat=None):
    if i == 1:
        return [max(omega_0, omega_p), min(torch.exp(torch.tensor(min(beta, 1), device="cuda")) * omega_p, omega_hat)]
    if i == 2:
        return [max(omega_0, 2 * torch.sqrt(torch.tensor(beta, device="cuda")), omega_hat - 2 * beta,
                    torch.exp(torch.tensor(beta, device="cuda")) * omega_p),
                omega_hat]
    if i == 3:
        return [max(omega_0, omega_hat - 2 * beta, torch.tensor(np.e, device="cuda") * omega_p),
                min(torch.exp(torch.tensor(beta, device="cuda")), omega_hat)]
    if i == 4:
        return [max(omega_0, omega_hat - 2 * beta, torch.tensor(np.e, device="cuda") * omega_p),
                min(torch.exp(torch.tensor(beta, device="cuda")) * omega_p, omega_hat,
                    (omega_hat / 4) * (1 + torch.sqrt(torch.tensor(1 - (32 / omega_hat ** 2), device="cuda"))))]
    if i == 5:
        return [max(omega_0, omega_hat - 2 * beta, torch.tensor(np.e, device="cuda") * omega_p,
                    (omega_hat / 4) * (1 - torch.sqrt(torch.tensor(1 - (32 / omega_hat ** 2), device="cuda")))),
                min(torch.exp(torch.tensor(beta, device="cuda")) * omega_p, omega_hat)]


def SOLVE_O1(omega_p, omega_hat, a1, b1, beta=1.0):
    omega_0 = b1 - a1
    S = [omega_0]
    if max(omega_0, omega_hat) < omega_p:
        return FIND_MIN(interval=[max(omega_0, omega_hat), omega_p], objective_func=xi_prime, omega_p=omega_p,
                        omega_hat=omega_hat, beta=beta)
    elif max(omega_0, omega_p) < omega_hat:
        S.append(omega_hat)
        S.append(FIND_MIN(J_getter(1, omega_0=omega_0, omega_p=omega_p, beta=beta, omega_hat=omega_hat),
                          objective_func=xi_prime, omega_p=omega_p, omega_hat=omega_hat, beta=beta))
        S.append(FIND_MIN(J_getter(2, omega_0=omega_0, omega_p=omega_p, beta=beta, omega_hat=omega_hat),
                          objective_func=xi_prime, omega_p=omega_p, omega_hat=omega_hat, beta=beta))
        if omega_hat <= 4 * torch.sqrt(torch.tensor(2., device="cuda")):
            S.append(FIND_MIN(J_getter(3, omega_0=omega_0, omega_p=omega_p, beta=beta, omega_hat=omega_hat),
                              objective_func=sigma, omega_p=omega_p, omega_hat=omega_hat, beta=beta))
        else:
            S.append(FIND_MIN(J_getter(4, omega_0=omega_0, omega_p=omega_p, beta=beta, omega_hat=omega_hat),
                              objective_func=sigma, omega_p=omega_p, omega_hat=omega_hat, beta=beta))
            S.append(FIND_MIN(
                J_getter(5, omega_0=omega_0, omega_p=omega_p, beta=beta, omega_hat=omega_hat),
                objective_func=sigma, omega_p=omega_p, omega_hat=omega_hat, beta=beta))
    return S[
        torch.argmin(torch.tensor([xi(omega=cur, omega_p=omega_p, omega_hat=omega_hat) for cur in S], device="cuda"))]


def SOLVE_O2(delta_p, omega_p, a2, b2, beta=1.0):
    omega_hat1 = 2 * (delta_p - a2)
    omega_hat2 = 2 * (b2 - delta_p)
    if omega_p >= max(omega_hat1, omega_hat2):
        return (delta_p, omega_p)
    omega_1 = SOLVE_O1(omega_p=omega_p, omega_hat=omega_hat1, a1=a2, b1=b2, beta=beta)
    omega_2 = SOLVE_O1(omega_p=omega_p, omega_hat=omega_hat2, a1=a2, b1=b2, beta=beta)
    # xi is the objective function of O1
    if xi(omega=omega_1, omega_p=omega_p, omega_hat=omega_hat1) <= \
            xi(omega=omega_2, omega_p=omega_p, omega_hat=omega_hat2):
        return (a2 + (omega_1 / 2), omega_1)
    else:
        return (b2 - (omega_2 / 2), omega_2)


def case_distinction(pred, proposal_list, cases, target, crop_shapes, axis=0, beta=1.):
    # depending on the axis we either optimize the target for top and bottom or left and right cuts
    # therefore depending on the axis the needed indices for coordinates and deltas are different
    if axis == 0:
        left_or_top, right_or_bottom = 0, 1
        delta_id, omega_id = 0, 2
    elif axis == 1:
        left_or_top, right_or_bottom = 2, 3
        delta_id, omega_id = 1, 3
    else:
        raise Exception('No proper axis selected, no axis with value: {}'.format(str(axis)))

    if not cases[left_or_top] and not cases[right_or_bottom]:
        # there was no cut on the current axis dimension
        # therefore we do not need to change the target for this dimension
        delta_label = target[delta_id]
        omega_label = target[omega_id]
    elif not cases[left_or_top] and cases[right_or_bottom]:
        # The gt was cut on the higher end of this axis (right or bottom)
        # as discussed and analogous to bbox2delta we extend the width by 1 and move the center by 0.5
        ca = 0.5 * (proposal_list[delta_id] + proposal_list[omega_id]) + 0.5
        da = proposal_list[omega_id] - proposal_list[delta_id] + 1
        a1 = target[delta_id] - 0.5 * target[omega_id]
        b1 = (crop_shapes[axis] - ca) / da
        omega_hat = 2 * (pred[delta_id] - a1)
        omega_star = SOLVE_O1(omega_p=pred[omega_id], omega_hat=omega_hat, a1=a1, b1=b1, beta=beta)
        delta_star = a1 + 0.5 * omega_star
        delta_label = delta_star
        omega_label = omega_star
    elif cases[left_or_top] and not cases[right_or_bottom]:
        # The gt was cut on the lower end of this axis (left or top)
        # as discussed and analogous to bbox2delta we extend the width by 1 and move the center by 0.5
        ca = 0.5 * (proposal_list[delta_id] + proposal_list[omega_id]) + 0.5
        da = proposal_list[omega_id] - proposal_list[delta_id] + 1
        a1 = - ca / da
        b1 = target[delta_id] + 0.5 * target[omega_id]
        omega_hat = 2 * (b1 - pred[delta_id])
        omega_star = SOLVE_O1(omega_p=pred[omega_id], omega_hat=omega_hat, a1=a1, b1=b1, beta=beta)
        delta_star = b1 - 0.5 * omega_star
        delta_label = delta_star
        omega_label = omega_star
    else:
        # gt was cut on both ends of the axis
        # as discussed and analogous to bbox2delta we extend the width by 1 and move the center by 0.5
        ca = 0.5 * (proposal_list[delta_id] + proposal_list[omega_id]) + 0.5
        da = proposal_list[omega_id] - proposal_list[delta_id] + 1
        a2 = - ca / da
        b2 = (crop_shapes[delta_id] - ca) / da
        delta_label, omega_label = SOLVE_O2(delta_p=pred[delta_id], omega_p=pred[omega_id], a2=a2, b2=b2, beta=beta)
    return delta_label, omega_label


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
                crop_info,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                plot=False,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        """
        cases:
        [lower x cropped?][upper x cropped?][lower y cropped?][upper y cropped]
        delta notation:
        dx, dy, dw, dh
        """

        target = bbox2delta(proposal_list, crop_info["orig_gt_left_top"])
        cases = crop_info["cases"]
        # we are doing exp as cabb does not use log scale
        # TODO remove once we do no longer plot we can remove the copy or move the reversion of this to plot
        pred[:, [2, 3]] = torch.exp(pred[:, [2, 3]])
        target[:, [2, 3]] = torch.exp(target[:, [2, 3]])
        loss = self.loss_weight * batched_bbox_loss(pred=pred, target=target, proposal_list=proposal_list,
                                                    cases=cases, crop_shapes=crop_shapes,
                                                    crop_info=crop_info, plot=plot, beta=self.beta, weight=weight,
                                                    reduction=reduction,
                                                    avg_factor=avg_factor,
                                                    **kwargs)
        return loss


def rect_crop_to_original(bbox0, bbox1, bbox2, bbox3, crop_left_x, crop_top_y, ec, ls):
    return patches.Rectangle((bbox0 + crop_left_x, bbox1 + crop_top_y),
                             bbox2 - bbox0, bbox3 - bbox1,
                             linewidth=1, edgecolor=ec,
                             linestyle=ls, facecolor='none')


def plot_anchors_and_gt(img, gt, anchor, label, prediction, crop_left_x, crop_top_y, wh, case):
    case = case.cpu()
    fig, ax = plt.subplots()
    ax.imshow(img[0].cpu().numpy())
    gt = delta2bbox(anchor.unsqueeze(0), gt.unsqueeze(0))[0]
    prediction = delta2bbox(anchor.unsqueeze(0), prediction.unsqueeze(0))[0]
    label = delta2bbox(anchor.unsqueeze(0), torch.tensor(label).unsqueeze(0))[0]
    bbox = anchor.cpu().numpy()
    ax.add_artist(
        rect_crop_to_original(bbox[0], bbox[1], bbox[2], bbox[3], crop_left_x, crop_top_y, ec="r", ls="-"))

    bbox = gt.cpu().numpy()
    ax.add_artist(
        rect_crop_to_original(bbox[0], bbox[1], bbox[2], bbox[3], crop_left_x, crop_top_y, ec="b", ls="-."))

    bbox = prediction.cpu().detach().numpy()
    ax.add_artist(
        rect_crop_to_original(bbox[0], bbox[1], bbox[2], bbox[3], crop_left_x, crop_top_y, ec="g", ls="--"))

    bbox = label.cpu().numpy()
    rect = rect_crop_to_original(bbox[0], bbox[1], bbox[2], bbox[3], crop_left_x, crop_top_y, ec="y", ls=":")
    ax.add_artist(rect)

    rect = patches.Rectangle((crop_left_x, crop_top_y),
                             wh[0].cuda(), wh[1].cuda(),
                             linewidth=1, edgecolor="c",
                             linestyle="-", facecolor='none')
    ax.add_artist(rect)

    rx, ry = rect.get_xy()
    cx = rx + rect.get_width() / 2.0
    cy = ry + rect.get_height() / 2.0
    text_case = "crop at: "
    if torch.equal(case, torch.tensor([0., 0., 0., 0.])):
        text_case += "none"
    else:
        if torch.equal(case[0], torch.tensor(1., )):
            text_case += "left "
        if torch.equal(case[1], torch.tensor(1.)):
            text_case += "right "
        if torch.equal(case[2], torch.tensor(1.)):
            text_case += "top "
        if torch.equal(case[3], torch.tensor(1.)):
            text_case += "bottom "

    ax.annotate(text_case, (cx, cy), color='m', weight='bold',
                fontsize=6, ha='center', va='center')

    plt.title("anchor in red, gt in blue, prediction in green, cabb target in yellow")
    plt.show()
