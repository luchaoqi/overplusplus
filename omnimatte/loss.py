import torch
import torch.nn as nn
import torch.nn.functional as F


class Loss:
    def __init__(self, config, mask_super=None):
        self.config =config
        self.weights = {
            "recon": config.omnimatte.loss_weight_recon,
            "alpha_reg_l0": config.omnimatte.loss_weight_alpha_reg_l0,
            "alpha_reg_l1": config.omnimatte.loss_weight_alpha_reg_l1,
            "mask_super": config.omnimatte.loss_weight_mask_super,
            "mask_super_1s": config.omnimatte.loss_weight_mask_super_ones,
            "smoothness": config.omnimatte.loss_weight_smoothness,
        }
        self.steps = {
            "alpha_reg_l0": config.omnimatte.loss_weight_alpha_reg_l0_steps,
            "alpha_reg_l1": config.omnimatte.loss_weight_alpha_reg_l1_steps,
            "mask_super": config.omnimatte.loss_weight_mask_super_steps,
            "mask_super_1s": config.omnimatte.loss_weight_mask_super_ones_steps,
            "smoothness": config.omnimatte.loss_weight_smoothness_steps,
        }
        self.gammas = {
            "alpha_reg_l0": config.omnimatte.loss_weight_alpha_reg_l0_gamma,
            "alpha_reg_l1": config.omnimatte.loss_weight_alpha_reg_l1_gamma,
            "mask_super": config.omnimatte.loss_weight_mask_super_gamma,
            "mask_super_1s": config.omnimatte.loss_weight_mask_super_ones_gamma,
            "smoothness": config.omnimatte.loss_weight_smoothness_gamma,
        }
        self.loss_recon_fn = {
            'l1': l1_loss,
            'l2': l2_loss,
        }[config.omnimatte.loss_recon_metric]
        self.loss_mask_super_fn = {
            'l1': l1_loss_masked,
            'l2': l2_loss_masked,
        }[config.omnimatte.loss_mask_super_metric]

        if mask_super is not None:
            self.mask_normalization = torch.mean(mask_super)
        else:
            self.mask_normalization = 1.
        for loss_term in ['alpha_reg_l0', 'alpha_reg_l1']:
            self.weights[loss_term] *= self.mask_normalization

        self.alpha_reg_l0_k = config.omnimatte.loss_weight_alpha_reg_l0_k


    def __call__(self, variables, constants, step_id):
        for key in self.weights:
            if key in self.steps and step_id in self.steps[key]:
                self.weights[key] *= self.gammas[key]
        
        XY_composited = variables['X_rgb'] * variables['X_alpha'] + constants['Y'] * (1 - variables['X_alpha'])
        losses = {}
        intermediates = {
            'composited': XY_composited,
        }
        losses['recon'], intermediates['recon_err'] = self.loss_recon_fn(constants['XY'], XY_composited)
        losses['alpha_reg_l0'] = sparsity_l0_loss(variables['X_alpha'], k=self.alpha_reg_l0_k)
        losses['alpha_reg_l1'] = sparsity_l1_loss(variables['X_alpha'])

        if 'mask' in constants and (self.weights['mask_super'] > 0 or self.weights['mask_super_1s'] > 0):
            mask_super_1s = self.loss_mask_super_fn(variables['X_alpha'], constants['mask'], constants['mask'])
            if self.weights['mask_super'] > 0:
                losses['mask_super'] = mask_super_1s
                losses['mask_super'] += self.loss_mask_super_fn(variables['X_alpha'], constants['mask'], 1. - constants['mask'])

            elif self.weights['mask_super_1s'] > 0:
                losses['mask_super_1s'] = mask_super_1s

        if self.weights['smoothness'] > 0:
            losses['smoothness'] = tv_loss(variables['X_alpha'])

        loss_total = 0.
        for key in losses:
            loss_total += losses[key] * self.weights[key]
        return loss_total, losses, intermediates


def l2_loss(pred, target):
    err = torch.square(pred - target)
    return torch.mean(err), err

def l1_loss(pred, target):
    err = torch.abs(pred - target)
    return torch.mean(err), err

def l2_loss_masked(pred, target, mask):
    return (torch.square(pred - target) * mask).sum() / (mask.sum() + 1e-9)

def l1_loss_masked(pred, target, mask):
    return (torch.abs(pred - target) * mask).sum() / (mask.sum() + 1e-9)

def sparsity_l1_loss(pred):
    return torch.mean(torch.abs(pred))

def sparsity_l0_loss(pred, k=5.0):
    return torch.mean((torch.sigmoid(pred * k) - 0.5) * 2.0)

def tv_loss(pred):
    grad_x = torch.abs(pred[..., 1:] - pred[..., :-1]).mean()
    grad_y = torch.abs(pred[..., 1:, :] - pred[..., :-1, :]).mean()
    return grad_x + grad_y