import torch
import torch.nn as nn
import numpy as np
from ..registry import LOSSES
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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


def FIND_MIN(interval, objective_func, omega_p, omega_hat, beta=1.0, eps_threshold=0.00001):
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
        S.append(omega_hat)
        S.append(FIND_MIN(J_getter(1, omega_0=omega_0, omega_p=omega_p, beta=beta, omega_hat=omega_hat),
                                 objective_func=epsilon_prime, omega_p=omega_p, omega_hat=omega_hat, beta=beta))
        S.append(FIND_MIN(J_getter(2, omega_0=omega_0, omega_p=omega_p, beta=beta, omega_hat=omega_hat),
                      objective_func=epsilon_prime, omega_p=omega_p, omega_hat=omega_hat, beta=beta))
        if omega_hat <= 4 * np.sqrt(2):
            S.append(FIND_MIN(J_getter(3, omega_0=omega_0, omega_p=omega_p, beta=beta, omega_hat=omega_hat),
                          objective_func=sigma, omega_p=omega_p, omega_hat=omega_hat, beta=beta))
        else:
            S.append(FIND_MIN(J_getter(4, omega_0=omega_0, omega_p=omega_p, beta=beta, omega_hat=omega_hat),
                          objective_func=sigma, omega_p=omega_p, omega_hat=omega_hat, beta=beta))
            S.append(FIND_MIN(
                J_getter(5, omega_0=omega_0, omega_p=omega_p, beta=beta, omega_hat=omega_hat),
                objective_func=sigma, omega_p=omega_p, omega_hat=omega_hat, beta=beta))
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


def case_distinction(i, pred, proposal_list, cases, target, crop_shapes, axis=0):


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
        print('DEFAULT CASE!')
        # x dimension has not been cut
        delta_label = target[i][delta_id]
        omega_label = target[i][omega_id]
    elif not cases[i][left_or_top] and cases[i][right_or_bottom]:
        print('RIGHT OR BOTTOM CUT')
        # x cut left --> O1 first case
        ca = 0.5 * (proposal_list[i][delta_id] + proposal_list[i][omega_id])
        da = proposal_list[i][omega_id] - proposal_list[i][delta_id]
        a1 = target[i][delta_id] - 0.5 * target[i][omega_id]
        b1 = (crop_shapes[i][axis] - ca) / da
        omega_hat = 2 * (pred[i][delta_id] - a1)
        omega_star = SOLVE_O1(omega_p=pred[i][omega_id], omega_hat=omega_hat, a1=a1, b1=b1)
        delta_star = a1 + 0.5 * omega_star
        delta_label = delta_star
        omega_label = omega_star
    elif cases[i][left_or_top] and not cases[i][right_or_bottom]:
        print('LEFT OR TOP CUT')
        # x cut right --> O1 second case
        ca = 0.5 * (proposal_list[i][delta_id] + proposal_list[i][omega_id])
        da = proposal_list[i][omega_id] - proposal_list[i][delta_id]
        a1 = - ca / da
        b1 = target[i][delta_id] + 0.5 * target[i][omega_id]
        omega_hat = 2 * (b1 - pred[i][delta_id])
        omega_star = SOLVE_O1(omega_p=pred[i][omega_id], omega_hat=omega_hat, a1=a1, b1=b1)
        delta_star = b1 - 0.5 * omega_star
        delta_label = delta_star
        omega_label = omega_star
    else:
        print('BOTH')
        # x cut on both sides --> O2
        ca = 0.5 * (proposal_list[i][delta_id] + proposal_list[i][omega_id])
        da = proposal_list[i][omega_id] - proposal_list[i][delta_id]
        a2 = - ca / da
        b2 = (crop_shapes[i][delta_id] - ca) / da
        delta_label, omega_label = SOLVE_O2(delta_p=pred[i][delta_id], omega_p=pred[i][delta_id], a2=a2, b2=b2)
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
                img,
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
        # assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        """
        cases:
        [lower x cropped?][upper x cropped?][lower y cropped?][upper y cropped]
        delta notation:
        dx, dy, dw, dh
        """


        # TODO: why are some ground truths = [0, 0, 0, 0] ?
        pred = pred.cpu().detach()
        target = target.cpu().detach()
        crop_shapes = crop_shapes.cpu().detach()
        proposal_list = proposal_list.cpu().detach()
        cases = cases.cpu().detach()

        for i in range(pred.shape[0]):
            print(f'Bbox prediction: {pred.cpu()[i]}')
            print(f'Associated Proposal: {proposal_list[i]}')
            print(f'Case for this prediction: {cases.cpu()[i]}')
            print(f'GT for this prediction: {target.cpu()[i]}')
            print(f'Crop dimensions: {crop_shapes[i]}')
            label = [None, None, None, None]
            target[i][[2,3]] = np.exp(target[i][[2,3]])
            pred[i][[2, 3]] = np.exp(pred[i][[2, 3]])
            # optimize in x
            label[0], label[2] = case_distinction(i, pred, proposal_list, cases, target, crop_shapes, axis=0)
            # optimize in y
            label[1], label[3] = case_distinction(i, pred, proposal_list, cases, target, crop_shapes, axis=1)
            print(f'New Target: {torch.tensor(label)}')
            self.plot_anchors_and_gt(img, target[i], proposal_list[i], label, pred[i])

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

    def transform_bbox_with_deltas(self, anchor, deltas):
        ca_x = 0.5 * (anchor[0] + anchor[2])
        da_x = anchor[2] - anchor[0]
        ca_y = 0.5 * (anchor[1] + anchor[3])
        da_y = anchor[3] - anchor[1]
        new_ca_x = ca_x + deltas[0] * da_x
        new_ca_y = ca_y + deltas[1] * da_y
        new_da_x = deltas[2] * da_x
        new_da_y = deltas[3] * da_y
        new_x_l = new_ca_x - 0.5 * new_da_x
        new_y_l = new_ca_y - 0.5 * new_da_y
        new_x_r = new_ca_x + 0.5 * new_da_x
        new_y_r = new_ca_y + 0.5 * new_da_y
        return torch.tensor([new_x_l, new_y_l, new_x_r, new_y_r])


    def plot_anchors_and_gt(self, img, gt, anchor, label, prediction):
        # Create figure and axes
        fig, ax = plt.subplots()
        _, c, x, y = img.shape
        img = img.permute(0, 2, 3, 1)[0].cpu().numpy()
        img += 1.5
        ax.imshow(img)
        gt = self.transform_bbox_with_deltas(anchor, gt)
        prediction = self.transform_bbox_with_deltas(anchor, prediction)
        label = self.transform_bbox_with_deltas(anchor, label)
        print(f'Gt bitches {gt}')
        print(f'Label bitches {label}')
        bbox = anchor.numpy()
        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], linewidth=1, edgecolor='b'
                                 , facecolor='none')
        ax.add_artist(rect)

        bbox = gt.numpy()
        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], linewidth=1, edgecolor='g'
                                 , facecolor='none')
        ax.add_artist(rect)

        bbox = prediction.numpy()
        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], linewidth=1, edgecolor='r'
                                 , facecolor='none')
        ax.add_artist(rect)

        bbox = label.numpy()
        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], linewidth=1, edgecolor=(0.5, 0.1, 0.3)
                                 , facecolor='none')
        ax.add_artist(rect)

        rx, ry = rect.get_xy()
        cx = rx + rect.get_width() / 2.0
        cy = ry + rect.get_height() / 2.0


        plt.show()