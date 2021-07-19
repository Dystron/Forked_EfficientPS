import torch
import torch.nn as nn
import numpy as np
from ..registry import LOSSES
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from .utils import weighted_loss
from mmdet.core import (bbox2result, bbox2roi, bbox_mapping, build_assigner,
                        build_sampler, multiclass_nms, delta2bbox, bbox2delta)
from PIL import Image


def xi(omega, omega_p, omega_hat, beta=1.0):
    x = (omega - omega_hat) / 2
    return bbox_loss(x, omega, omega_p)


def xi_prime(omega, omega_p, omega_hat, beta=1.0):
    x = (omega - omega_hat) / 2
    diff = np.log(omega) - np.log(omega_p)
    return 0.5 * smooth_l1_loss_prime(x, beta) + smooth_l1_loss_prime(diff, beta) / omega


# phi is objective function of O2 which we never use
# def phi(omega, omega_p, delta, delta_p, a1, beta=1.0):
#     return smooth_l1_loss(delta - delta_p, beta) + smooth_l1_loss(np.log(omega) - np.log(omega_p))


def sigma(omega, omega_p, omega_hat, beta=1.0):
    return omega * xi_prime(omega, omega_p, omega_hat, beta)


def bbox_loss(x, omega, omega_p, beta=1.0):
    diff = np.log(omega_p) - np.log(omega)
    return smooth_l1_loss(x, beta) + smooth_l1_loss(diff, beta)

@weighted_loss
def smooth_l1_loss(x, beta=1.0):
    assert beta > 0
    diff = torch.abs(x)
    loss = torch.where(diff < beta, (diff * diff) / (beta * 2), diff - 0.5 * beta)
    return loss


def smooth_l1_loss_prime(x, beta=1.0):
    return np.clip(x / beta, a_min=-1, a_max=1)


def FIND_MIN(interval, objective_func, omega_p, omega_hat, beta=1.0, eps_threshold=0.1):
    # interval is min and max for omega
    u, v = interval
    if objective_func(omega=u, omega_p=omega_p, omega_hat=omega_hat,
                      beta=beta) >= 0:
        return u
    elif objective_func(v, omega_p=omega_p, omega_hat=omega_hat,
                        beta=beta) <= 0:
        return v
    else:
        m = (u + v) / 2
        if v - u < eps_threshold:
            return m
        elif objective_func(m, omega_p=omega_p, omega_hat=omega_hat,
                            beta=beta) >= 0:
            return FIND_MIN([u, m], objective_func, omega_p, omega_hat, beta)
        else:
            return FIND_MIN([m, v], objective_func, omega_p, omega_hat, beta)


def J_getter(i, omega_0=None, omega_p=None, beta=None, omega_hat=None):
    if i == 1:
        return [max(omega_0, omega_p), min(np.e ** min(beta, 1) * omega_p, omega_hat)]
    if i == 2:
        return [max(omega_0, 2 * np.sqrt(beta), omega_hat - 2 * beta, (np.e ** beta) * omega_p), omega_hat]
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
        return FIND_MIN(interval=[max(omega_0, omega_hat), omega_p], objective_func=xi_prime, omega_p=omega_p,
                        omega_hat=omega_hat, beta=beta)
    elif max(omega_0, omega_p) < omega_hat:
        S.append(omega_hat)
        S.append(FIND_MIN(J_getter(1, omega_0=omega_0, omega_p=omega_p, beta=beta, omega_hat=omega_hat),
                          objective_func=xi_prime, omega_p=omega_p, omega_hat=omega_hat, beta=beta))
        S.append(FIND_MIN(J_getter(2, omega_0=omega_0, omega_p=omega_p, beta=beta, omega_hat=omega_hat),
                          objective_func=xi_prime, omega_p=omega_p, omega_hat=omega_hat, beta=beta))
        if omega_hat <= 4 * np.sqrt(2):
            S.append(FIND_MIN(J_getter(3, omega_0=omega_0, omega_p=omega_p, beta=beta, omega_hat=omega_hat),
                              objective_func=sigma, omega_p=omega_p, omega_hat=omega_hat, beta=beta))
        else:
            S.append(FIND_MIN(J_getter(4, omega_0=omega_0, omega_p=omega_p, beta=beta, omega_hat=omega_hat),
                              objective_func=sigma, omega_p=omega_p, omega_hat=omega_hat, beta=beta))
            S.append(FIND_MIN(
                J_getter(5, omega_0=omega_0, omega_p=omega_p, beta=beta, omega_hat=omega_hat),
                objective_func=sigma, omega_p=omega_p, omega_hat=omega_hat, beta=beta))
    return S[np.argmin([xi(omega=cur, omega_p=omega_p, omega_hat=omega_hat, beta=beta) for cur in S])]


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
    if xi(omega=omega_1, omega_p=omega_p, omega_hat=omega_hat1, beta=beta) <= \
            xi(omega=omega_2, omega_p=omega_p, omega_hat=omega_hat2, beta=beta):
        return (a2 + (omega_1 / 2), omega_1)
    else:
        return (b2 + (omega_2 / 2), omega_2)


def case_distinction(i, pred, proposal_list, cases, target, crop_shapes, axis=0):
    # if i == 12:
    #     log_pred = pred.clone()
    #     log_target = target.clone()
    #     log_pred[:, [2, 3]] = np.log(log_pred[:, [2, 3]])
    #     log_target[:, [2, 3]] = np.log(log_target[:, [2, 3]])
    #     pred_top_left = delta2bbox(proposal_list[i].unsqueeze(0), log_pred[i].unsqueeze(0))
    #     target_top_left = delta2bbox(proposal_list[i].unsqueeze(0), log_target[i].unsqueeze(0))
    #     print(f'We are at case distinction of prediction 12, analyzing axis {axis}\n'
    #           f'pred = {pred[i]}\n'
    #           f'pred_top_left = {pred_top_left}\n'
    #           f'proposal_list = {proposal_list[i]}\n'
    #           f'cases = {cases[i]}\n'
    #           f'target = {target[i]}\n'
    #           f'target_top_left = {target_top_left}\n'
    #           f'crop_shapes of that axis = {crop_shapes[i][axis]}\n')
    # if axis = 0 optimize x coordinate else y
    if axis == 0:
        left_or_top, right_or_bottom = 0, 1
        delta_id, omega_id = 0, 2
    elif axis == 1:
        left_or_top, right_or_bottom = 2, 3
        delta_id, omega_id = 1, 3
    else:
        raise Exception('No proper axis selected, no axis with value: {}'.format(str(axis)))

    label = [None, None, None, None]
    # set the x dimension parameters of the target label
    if not cases[i][left_or_top] and not cases[i][right_or_bottom]:
        # print('DEFAULT CASE!')
        # x dimension has not been cut
        delta_label = target[i][delta_id]
        omega_label = target[i][omega_id]
    elif not cases[i][left_or_top] and cases[i][right_or_bottom]:
        # print('RIGHT OR BOTTOM CUT')
        # x cut left --> O1 first case
        # TODO correct like that?
        # + 1 is analogous to bbox2delta
        # removed + 0.5 here as I think its wrong, we just have a center here.
        ca = 0.5 * (proposal_list[i][delta_id] + proposal_list[i][omega_id])
        # removed + 1 here as I think its wrong, we just have a distance here.
        da = proposal_list[i][omega_id] - proposal_list[i][delta_id]
        a1 = target[i][delta_id] - 0.5 * target[i][omega_id]
        b1 = (crop_shapes[i][axis] - ca) / da
        omega_hat = 2 * (pred[i][delta_id] - a1)
        omega_star = SOLVE_O1(omega_p=pred[i][omega_id], omega_hat=omega_hat, a1=a1, b1=b1)
        delta_star = a1 + 0.5 * omega_star
        delta_label = delta_star
        omega_label = omega_star
    elif cases[i][left_or_top] and not cases[i][right_or_bottom]:
        # print('LEFT OR TOP CUT')
        # x cut right --> O1 second case
        ca = 0.5 * (proposal_list[i][delta_id] + proposal_list[i][omega_id]) + 0.5
        da = proposal_list[i][omega_id] - proposal_list[i][delta_id] + 1
        a1 = - ca / da
        b1 = target[i][delta_id] + 0.5 * target[i][omega_id]
        omega_hat = 2 * (b1 - pred[i][delta_id])
        omega_star = SOLVE_O1(omega_p=pred[i][omega_id], omega_hat=omega_hat, a1=a1, b1=b1)
        delta_star = b1 - 0.5 * omega_star
        delta_label = delta_star
        omega_label = omega_star
    else:
        # print('BOTH')
        # x cut on both sides --> O2
        ca = 0.5 * (proposal_list[i][delta_id] + proposal_list[i][omega_id]) + 0.5
        da = proposal_list[i][omega_id] - proposal_list[i][delta_id] - 1
        a2 = - ca / da
        b2 = (crop_shapes[i][delta_id] - ca) / da
        delta_label, omega_label = SOLVE_O2(delta_p=pred[i][delta_id], omega_p=pred[i][omega_id], a2=a2, b2=b2)
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
                img_metas,
                pred,
                target,
                crop_shapes,
                proposal_list,
                sampling_results,
                crop_info,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                plot=False,
                **kwargs):
        # assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        """
        cases:
        [lower x cropped?][upper x cropped?][lower y cropped?][upper y cropped]
        delta notation:
        dx, dy, dw, dh
        """

        pred = pred.cpu().detach()
        target = target.cpu().detach()
        # print(f'target shape cabb\n{target.shape}')
        crop_shapes = crop_shapes.cpu().detach()
        proposal_list = proposal_list.cpu().detach()
        orig_target = bbox2delta(proposal_list, crop_info["orig_gt_left_top"])
        cases = crop_info["cases"].cpu().detach()
        # we are doing exp as cabb does not use log scale
        # todo once we do no longer plot we can remove the copy or move the reversion of this to plot
        pred_copy = pred.clone()
        pred_copy[:, [2, 3]] = np.exp(pred_copy[:, [2, 3]])
        target_copy = orig_target.clone()
        target_copy[:, [2, 3]] = np.exp(target_copy[:, [2, 3]])

        # assert pred.shape == orig_target.shape
        # assert proposal_list.shape == orig_target.shape
        # assert crop_shapes.shape[0] == orig_target.shape[0]
        # assert pred.shape[0] == orig_target.shape[0]
        loss = 0
        for i in range(pred.shape[0]):
            label = [None, None, None, None]
            # optimize in x
            label[0], label[2] = case_distinction(i, pred_copy, proposal_list, cases, target_copy, crop_shapes, axis=0)
            # optimize in y
            label[1], label[3] = case_distinction(i, pred_copy, proposal_list, cases, target_copy, crop_shapes, axis=1)
            label = torch.tensor(label)
            label = np.array(label)
            x = pred_copy[i][[0, 1]] - label[[0, 1]]
            # we dont need to do log of label even tough it is not in log notation
            # as there is a log inside the loss function
            i_loss = bbox_loss(x, label[[2, 3]], pred_copy[i][[2, 3]])
            loss += i_loss
            if plot:
                # we log the label only for the plotting og the loss
                label[[2, 3]] = np.log(label[[2, 3]])
                self.plot_anchors_and_gt(crop_info["orig_image"], orig_target[i], proposal_list[i], label, pred[i],
                                         *crop_info["crop_left_top"], crop_shapes[i], crop_info["cases"][i])
        loss /= pred_copy.shape[0]

        # todo MORE EFFICIENT SLICE IN X DIM -->4 CASES THEN CONCAT THEN SLICE IN Y DIM --> 4 CASES

        """loss_bbox = self.loss_weight * smooth_l1_loss(
            pred,
            target,
            weight,
            beta=self.beta,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss_bbox"""
        # todo do we want to multiply this with a loss weight as in smoothl1loss?
        return loss

    def rect_crop_to_original(self, bbox0, bbox1, bbox2, bbox3, crop_left_x, crop_top_y, ec, ls):
        return patches.Rectangle((bbox0 + crop_left_x, bbox1 + crop_top_y),
                                 bbox2 - bbox0, bbox3 - bbox1,
                                 linewidth=1, edgecolor=ec,
                                 linestyle=ls, facecolor='none')

    def plot_anchors_and_gt(self, img, gt, anchor, label, prediction, crop_left_x, crop_top_y, wh, case):
        # Create figure and axes
        # print(f"crop_left_x crop_top_y\n"
        #       f"{crop_left_x, crop_top_y, crop_left_x + wh[0], crop_top_y + wh[1]}")
        fig, ax = plt.subplots()
        ax.imshow(img[0].cpu().numpy())
        # we unsqueeze as a 2d tensor is expercted and take the 0st elemt as it returns a list
        gt = delta2bbox(anchor.unsqueeze(0), gt.unsqueeze(0))[0]
        prediction = delta2bbox(anchor.unsqueeze(0), prediction.unsqueeze(0))[0]
        label = delta2bbox(anchor.unsqueeze(0), torch.tensor(label).unsqueeze(0))[0]
        # print(f'coord Gt {gt}')
        # print(f'coord pred {prediction}')
        # print(f'coord cabb target  {label}')
        bbox = anchor.numpy()
        ax.add_artist(self.rect_crop_to_original(bbox[0], bbox[1], bbox[2], bbox[3], crop_left_x, crop_top_y, ec="r", ls="-"))

        bbox = gt.numpy()
        ax.add_artist(self.rect_crop_to_original(bbox[0], bbox[1], bbox[2], bbox[3], crop_left_x, crop_top_y, ec="b", ls="-."))

        bbox = prediction.numpy()
        ax.add_artist(self.rect_crop_to_original(bbox[0], bbox[1], bbox[2], bbox[3], crop_left_x, crop_top_y, ec="g", ls="--"))

        bbox = label.numpy()
        rect = self.rect_crop_to_original(bbox[0], bbox[1], bbox[2], bbox[3], crop_left_x, crop_top_y, ec="y", ls=":")
        ax.add_artist(rect)

        rect = patches.Rectangle((crop_left_x, crop_top_y),
                                 wh[0].cuda(), wh[1].cuda(),
                                 linewidth=1, edgecolor="c",
                                 linestyle="-", facecolor='none')
        ax.add_artist(rect)

        rx, ry = rect.get_xy()
        cx = rx + rect.get_width() / 2.0
        cy = ry + rect.get_height() / 2.0
        # [lower x cropped?][upper x cropped?][lower y cropped?][upper y cropped]
        text_case = "crop at: "
        if torch.equal(case, torch.tensor([0.,0.,0.,0.])):
            text_case += "none"
        else:
            if torch.equal(case[0], torch.tensor(1.)):
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
