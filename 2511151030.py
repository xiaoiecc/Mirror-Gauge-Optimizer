# -*- coding: utf-8 -*-
"""
CIFAR-100
模型结构：
  ResNet-18 backbone (AdamW)
  → SwiGLU block (Mirror-Gauge 完全控制)
  → Linear classifier head (AdamW)

优化器：
  Mirror-Gauge（SwiGLU块用协变镜像下降+自适应规范变换+Dual Momentum+EMA；骨干与分类头用 AdamW）
  **修正版4：修复塌缩问题 (Fail-Safe, 稳定方向, 尺度夹紧)**
"""

import os
import math
import time
from dataclasses import dataclass
from typing import Tuple, Dict, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np


# ===========================
# §0. 模型组件
# ===========================

class ResNet18Backbone(nn.Module):
    """
    ResNet-18 backbone，输出全局平均池化后的特征（维度=512）
    """
    def __init__(self, pretrained: bool = False):
        super().__init__()
        model = torchvision.models.resnet18(weights=None if not pretrained else torchvision.models.ResNet18_Weights.DEFAULT)
        # 替换fc为Identity，使forward输出特征
        model.fc = nn.Identity()
        self.backbone = model
        self.out_features = 512

    def forward(self, x):
        return self.backbone(x)  # [B, 512]


class SwiGLUBlock(nn.Module):
    """
    SwiGLU(x) = (SiLU(x W1^T) ⊙ (x W2^T)) W3^T
    """
    def __init__(self, in_features: int, hidden_features: int, out_features: int):
        super().__init__()
        self.W1 = nn.Linear(in_features, hidden_features, bias=False)
        self.W2 = nn.Linear(in_features, hidden_features, bias=False)
        self.W3 = nn.Linear(hidden_features, out_features, bias=False)
        nn.init.kaiming_uniform_(self.W1.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W2.weight, a=math.sqrt(5))
        nn.init.xavier_uniform_(self.W3.weight)

    def forward(self, x):
        a = self.W1(x)           # [B, H]
        b = self.W2(x)           # [B, H]
        h = F.silu(a) * b        # [B, H]
        return self.W3(h)        # [B, out_features]


class ResNet18_SwiGLU_Classifier(nn.Module):
    """
    主模型：ResNet18 Backbone → SwiGLU Block → Linear Classifier
    """
    def __init__(self, num_classes: int = 100, hidden_features: int = 1024, pretrained_backbone: bool = False):
        super().__init__()
        self.backbone = ResNet18Backbone(pretrained=pretrained_backbone)
        feat_dim = self.backbone.out_features
        self.swiglu = SwiGLUBlock(in_features=feat_dim, hidden_features=hidden_features, out_features=feat_dim)
        self.classifier = nn.Linear(feat_dim, num_classes, bias=True)

    def forward(self, x):
        f = self.backbone(x)
        s = self.swiglu(f)
        logits = self.classifier(s)
        return logits


# ===========================
# §1. ψ的梯度与径向逆映射（修正版4）
# ===========================

@torch.no_grad()
def row_norms(W: torch.Tensor) -> torch.Tensor:
    return torch.norm(W, p=2, dim=1)

@torch.no_grad()
def col_norms(W: torch.Tensor) -> torch.Tensor:
    return torch.norm(W, p=2, dim=0)

@torch.no_grad()
def _log_row_col_barrier_grad(W: torch.Tensor, lam_rc: float = 1e-3, eps: float = 1e-6):
    R, C = W.shape
    r = torch.clamp(row_norms(W), min=eps).unsqueeze(1)
    c = torch.clamp(col_norms(W), min=eps).unsqueeze(0)
    grad = W / (r*r) + W / (c*c)
    return lam_rc * grad


def psi_nabla_block(W1: torch.Tensor, W2: torch.Tensor, W3: torch.Tensor):
    eps = 1e-12
    n1_2 = (W1 * W1).sum()
    n2_2 = (W2 * W2).sum()
    n3_2 = (W3 * W3).sum()
    
    lnP = (torch.log(n1_2 + eps) + torch.log(n2_2 + eps) + torch.log(n3_2 + eps)) / 3.0
    P = torch.exp(lnP)
    eps_norm = 1e-6 * torch.clamp(P.detach(), min=1.0)
    denom = P + eps_norm
    Ssum = n1_2 + n2_2 + n3_2

    def grad_first(W, nW2):
        term1 = W / denom
        term2 = (Ssum * P) / (3.0 * (denom * denom) + 1e-18) * (W / (nW2 + 1e-18))
        g = term1 - term2
        return torch.nan_to_num(g, nan=0.0, posinf=1e6, neginf=-1e6)

    g1 = grad_first(W1, n1_2)
    g2 = grad_first(W2, n2_2)
    g3 = grad_first(W3, n3_2)
    
    # 轻谱张力
    g2 = g2 + _log_row_col_barrier_grad(W2, lam_rc=1e-3)
    g3 = g3 + _log_row_col_barrier_grad(W3, lam_rc=1e-3)

    return g1, g2, g3


@torch.no_grad()
def md_inverse_map_block_radial(W1: torch.Tensor, W2: torch.Tensor, W3: torch.Tensor,
                                eta1: torch.Tensor, eta2: torch.Tensor, eta3: torch.Tensor,
                                iters=8, tau=0.3,
                                s_min=1e-4, s_max=100.0,      # s_min 抬高到 1e-4
                                alpha_floor=5e-6, eps_norm_factor=1e-6, # alpha_floor 抬高
                                rel_clip=0.15,              
                                dual_clip=0.25,             
                                verbose=False):
    
    def _dir_from_weight(W):
        n = W.norm()
        if n > 1e-8:
            return W / n
        else:
            # FIX: 使用稳定的确定性方向，避免随机噪声导致塌缩
            v = torch.ones_like(W)
            return v / (v.norm() + 1e-12)

    v1 = _dir_from_weight(W1)
    v2 = _dir_from_weight(W2)
    v3 = _dir_from_weight(W3)
    
    def proj_scalar(eta, v):
        return float(torch.sum(eta * v))

    e1 = proj_scalar(eta1, v1)
    e2 = proj_scalar(eta2, v2)
    e3 = proj_scalar(eta3, v3)

    s1 = max(float(W1.norm()), s_min) # 使用 s_min
    s2 = max(float(W2.norm()), s_min)
    s3 = max(float(W3.norm()), s_min)

    def alpha_i_func(s1_val, s2_val, s3_val, si_val):
        lnP = (2.0/3.0) * (math.log(max(s1_val, s_min)) +
                           math.log(max(s2_val, s_min)) +
                           math.log(max(s3_val, s_min)))
        P = math.exp(lnP)
        denom = P + max(1.0, P) * eps_norm_factor
        Ssum = s1_val*s1_val + s2_val*s2_val + s3_val*s3_val
        common = (Ssum * P) / (3.0 * (denom*denom) + 1e-18)
        
        si_safe = max(si_val, s_min)
        alpha_val = (si_safe / denom) - (common / si_safe)
        return alpha_val, denom

    s1_init, s2_init, s3_init = s1, s2, s3

    a1, _ = alpha_i_func(s1, s2, s3, s1)
    a2, _ = alpha_i_func(s1, s2, s3, s2)
    a3, _ = alpha_i_func(s1, s2, s3, s3)

    def clamp_target(ei, ai):
        base = max(abs(ai), alpha_floor)
        delta = max(min(ei - ai, dual_clip * base), -dual_clip * base)
        return ai + delta

    e1 = clamp_target(e1, a1)
    e2 = clamp_target(e2, a2)
    e3 = clamp_target(e3, a3)

    for t in range(iters):
        a1, _ = alpha_i_func(s1, s2, s3, s1)
        a2, _ = alpha_i_func(s1, s2, s3, s2)
        a3, _ = alpha_i_func(s1, s2, s3, s3)

        def update_radius(si, ai, ei):
            if abs(ai) < alpha_floor:
                ai = math.copysign(alpha_floor, ai)
            
            r = ei / ai
            
            r = max(min(r, 1.0 + rel_clip), 1.0 - rel_clip)
            target = si * r

            s_new = (1 - tau) * si + tau * target
            return min(max(s_new, s_min), s_max)

        s1_new = update_radius(s1, a1, e1)
        s2_new = update_radius(s2, a2, e2)
        s3_new = update_radius(s3, a3, e3)

        if verbose and t % 2 == 0:
            print(f"[Radial-Inv] iter {t}: s=({s1:.3e},{s2:.3e},{s3:.3e}), "
                  f"α=({a1:.3e},{a2:.3e},{a3:.3e}), e=({e1:.3e},{e2:.3e},{e3:.3e})")

        if max(abs(s1_new - s1), abs(s2_new - s2), abs(s3_new - s3)) < 1e-6:
            s1, s2, s3 = s1_new, s2_new, s3_new
            if verbose:
                print(f"[Radial-Inv] converged at iter {t}")
            break
        
        s1, s2, s3 = s1_new, s2_new, s3_new

    W1_new = s1 * v1
    W2_new = s2 * v2
    W3_new = s3 * v3

    if not (torch.isfinite(W1_new).all() and torch.isfinite(W2_new).all() and torch.isfinite(W3_new).all()):
        if verbose:
            print("[Radial-Inv] non-finite result detected -> fallback to old weights")
        return W1.detach(), W2.detach(), W3.detach()
    
    if verbose:
        rel1 = s1 / s1_init
        rel2 = s2 / s2_init
        rel3 = s3 / s3_init
        print(f"[Radial-Inv] final radius ratio: ({rel1:.3f}, {rel2:.3f}, {rel3:.3f})")
    
    return W1_new, W2_new, W3_new


# ===========================
# §2. 规范变换（低开销）+ 并行移动（修正版4）
# ===========================

@torch.no_grad()
def left_scale_log_inplace(W: torch.Tensor, logd: torch.Tensor):
    W.mul_(torch.exp(logd).unsqueeze(1))

@torch.no_grad()
def right_scale_log_inplace(W: torch.Tensor, logd: torch.Tensor):
    W.mul_(torch.exp(logd).unsqueeze(0))

@torch.no_grad()
def pt_left_scale_momentum_log(m: torch.Tensor, logd: torch.Tensor):
    m.mul_(torch.exp(-logd).unsqueeze(1))

@torch.no_grad()
def pt_right_scale_momentum_log(m: torch.Tensor, logd: torch.Tensor):
    m.mul_(torch.exp(-logd).unsqueeze(0))


@dataclass
class GaugeConfig:
    eta_g21: float = 0.05      
    eta_g22: float = 0.03      
    eta_g31: float = 0.10      
    eta_g32: float = 0.02      
    clip_channel_scale: float = 0.15  
    ema_decay: float = 0.9     
    pressure_w_triplet: float = 0.3
    pressure_w_mismatch: float = 0.5
    pressure_w_nonlinear: float = 0.2
    pressure_threshold: float = 0.08  
    md_full_interval: int = 10        
    warmup_steps: int = 400    
    warmup_g31_eta: float = 0.02  
    head_lr_factor: float = 0.03   # 进一步降低对偶学习率


class MirrorGaugeController:
    def __init__(self, block: SwiGLUBlock, lr: float, weight_decay: float,
                 beta_momentum: float = 0.0,
                 config: GaugeConfig = GaugeConfig()):
        self.block = block
        self.lr = lr
        self.wd = weight_decay
        self.beta = beta_momentum
        self.cfg = config
        self.head_lr = lr * config.head_lr_factor

        W1, W2, W3 = block.W1.weight, block.W2.weight, block.W3.weight
        self.m1 = torch.zeros_like(W1)
        self.m2 = torch.zeros_like(W2)
        self.m3 = torch.zeros_like(W3)

        H = W1.shape[0]
        device = W1.device
        self.ema_coef = torch.zeros(H, device=device)
        self.ema_zeta = torch.zeros(H, device=device)
        self.ema_ratio = torch.zeros(H, device=device)
        self.ema_silu_prime = torch.ones(H, device=device) * 0.5
        self.step_idx = 0
        
        self.momentum_off_steps = 0
        
        # 径向逆映射参数覆盖
        self._md_rel_clip_override = None
        self._md_dual_clip_override = None
        self._md_iters_override = None
        self._md_tau_override = None

    @torch.no_grad()
    def _update_ema_stats(self, x_in: torch.Tensor):
        W1, W2, W3 = self.block.W1.weight, self.block.W2.weight, self.block.W3.weight
        a = x_in @ W1.t()
        b = x_in @ W2.t()
        sigma = torch.sigmoid(a)
        silu_prime = sigma * (1 + a*(1 - sigma))
        silu_a = F.silu(a)

        coef = (1.0 - silu_prime).mean(dim=0)
        num = (a * silu_prime).mean(dim=0)
        den = silu_a.mean(dim=0) + 1e-8
        zeta = num / den
        ratio = (b / (a.abs() + 1e-5)).mean(dim=0)
        silu_p = silu_prime.mean(dim=0)

        d = self.cfg.ema_decay
        self.ema_coef.mul_(d).add_((1-d) * coef)
        self.ema_zeta.mul_(d).add_((1-d) * zeta)
        self.ema_ratio.mul_(d).add_((1-d) * ratio)
        self.ema_silu_prime.mul_(d).add_((1-d) * silu_p)

    @torch.no_grad()
    def _gauge_pressure(self) -> float:
        W1, W2, W3 = self.block.W1.weight, self.block.W2.weight, self.block.W3.weight
        norms = torch.tensor([W1.norm(), W2.norm(), W3.norm()], device=W1.device)
        triplet = torch.std(torch.log(norms + 1e-8)).item()
        w2_r = row_norms(W2)
        w3_c = col_norms(W3)
        mismatch = torch.mean(torch.abs(torch.log(w2_r + 1e-8) - torch.log(w3_c + 1e-8))).item()
        nonlinear = torch.mean(torch.abs(self.ema_coef)).item()
        s = (self.cfg.pressure_w_triplet * triplet +
             self.cfg.pressure_w_mismatch * mismatch +
             self.cfg.pressure_w_nonlinear * nonlinear)
        return s

    @torch.no_grad()
    def _apply_G31_triplet_balance(self, eta=None):
        W1, W2, W3 = self.block.W1.weight, self.block.W2.weight, self.block.W3.weight
        if eta is None: eta = self.cfg.eta_g31
            
        eps = 1e-12
        ln1 = torch.log(W1.norm() + eps)
        ln2 = torch.log(W2.norm() + eps)
        ln3 = torch.log(W3.norm() + eps)
        lroot = (ln1 + ln2 + ln3) / 3.0

        s1 = eta * (lroot - ln1)
        s2 = eta * (lroot - ln2)
        s3 = eta * (lroot - ln3)

        clip = self.cfg.clip_channel_scale
        s1 = torch.clamp(s1, -clip, clip)
        s2 = torch.clamp(s2, -clip, clip)
        s3 = torch.clamp(s3, -clip, clip)

        logd1 = W1.new_full((W1.shape[0],), float(s1))
        logd2 = W2.new_full((W2.shape[0],), float(s2))
        logd3 = W3.new_full((W3.shape[0],), float(s3))

        left_scale_log_inplace(W1, logd1)
        left_scale_log_inplace(W2, logd2)
        left_scale_log_inplace(W3, logd3)

        pt_left_scale_momentum_log(self.m1, logd1)
        pt_left_scale_momentum_log(self.m2, logd2)
        pt_left_scale_momentum_log(self.m3, logd3)

    @torch.no_grad()
    def _apply_G32_channel_output_coupling(self):
        W2, W3 = self.block.W2.weight, self.block.W3.weight
        eps = 1e-8
        
        w2_r = row_norms(W2)
        w3_c = col_norms(W3)
        
        log_ratio = torch.log(w2_r + eps) - torch.log(w3_c + eps)
        
        log_scale = torch.clamp(
            self.cfg.eta_g32 * log_ratio,
            -self.cfg.clip_channel_scale,
            self.cfg.clip_channel_scale
        )
        
        left_scale_log_inplace(W2, -log_scale)
        right_scale_log_inplace(W3, +log_scale)

        pt_left_scale_momentum_log(self.m2, -log_scale)
        pt_right_scale_momentum_log(self.m3, +log_scale)

    @torch.no_grad()
    def _apply_G21_silu_local_linearize(self):
        W1 = self.block.W1.weight
        t = self.cfg.eta_g21 * self.ema_coef
        t = torch.clamp(t, min=-0.95, max=10.0)
        log_scale_raw = torch.log1p(t)
        
        log_scale = torch.clamp(
            log_scale_raw,
            -self.cfg.clip_channel_scale,
            self.cfg.clip_channel_scale
        )
        left_scale_log_inplace(W1, log_scale)
        pt_left_scale_momentum_log(self.m1, log_scale)

    @torch.no_grad()
    def _apply_G22_channel_compensation(self):
        W1, W2, W3 = self.block.W1.weight, self.block.W2.weight, self.block.W3.weight
        eps = 1e-6

        t = self.cfg.eta_g22 * self.ema_zeta
        t = torch.clamp(t, min=-0.95, max=10.0)
        log_s_raw = torch.log1p(t)
        log_s = torch.clamp(
            log_s_raw,
            -self.cfg.clip_channel_scale,
            self.cfg.clip_channel_scale
        )

        left_scale_log_inplace(W2, -log_s)
        right_scale_log_inplace(W3, +log_s)

        pt_left_scale_momentum_log(self.m2, -log_s)
        pt_right_scale_momentum_log(self.m3, +log_s)

        c = - self.ema_zeta * (self.ema_ratio / (self.ema_silu_prime + eps))
        tc = self.cfg.eta_g22 * c
        tc = torch.clamp(tc, min=-0.95, max=10.0)
        log_lam_raw = torch.log1p(tc)
        log_lam = torch.clamp(
            log_lam_raw,
            -self.cfg.clip_channel_scale,
            self.cfg.clip_channel_scale
        )

        left_scale_log_inplace(W1, log_lam)
        pt_left_scale_momentum_log(self.m1, log_lam)

    @torch.no_grad()
    def _safe_apply_gauge(self):
        """应用所有规范变换（非 Fail-Safe 模式）"""
        W1, W2, W3 = self.block.W1.weight, self.block.W2.weight, self.block.W3.weight
        
        W1_backup = W1.clone(); W2_backup = W2.clone(); W3_backup = W3.clone()
        m1_backup = self.m1.clone(); m2_backup = self.m2.clone(); m3_backup = self.m3.clone()
        
        if self.step_idx < self.cfg.warmup_steps:
            # Warmup 期间，隔步执行弱 G31
            if (self.step_idx % 2) == 0:
                self._apply_G31_triplet_balance(eta=0.5 * self.cfg.warmup_g31_eta)
        else:
            # 正常模式
            self._apply_G31_triplet_balance()
            self._apply_G32_channel_output_coupling()
            self._apply_G21_silu_local_linearize()
            self._apply_G22_channel_compensation()
        
        if not (torch.isfinite(W1).all() and torch.isfinite(W2).all() and torch.isfinite(W3).all()):
            W1.copy_(W1_backup); W2.copy_(W2_backup); W3.copy_(W3_backup)
            self.m1.copy_(m1_backup); self.m2.copy_(m2_backup); self.m3.copy_(m3_backup)
            return False
        return True
    
    @torch.no_grad()
    def _apply_gauge_fail_safe(self):
        """高压模式下，仅应用极弱 G31"""
        W1, W2, W3 = self.block.W1.weight, self.block.W2.weight, self.block.W3.weight
        W1_backup = W1.clone(); W2_backup = W2.clone(); W3_backup = W3.clone()
        m1_backup = self.m1.clone(); m2_backup = self.m2.clone(); m3_backup = self.m3.clone()
        
        # 极轻的 G31
        eta_fs = min(0.5 * self.cfg.warmup_g31_eta, 0.01)
        # self._apply_G31_triplet_balance(eta=eta_fs)
        
        if not (torch.isfinite(W1).all() and torch.isfinite(W2).all() and torch.isfinite(W3).all()):
            W1.copy_(W1_backup); W2.copy_(W2_backup); W3.copy_(W3_backup)
            self.m1.copy_(m1_backup); self.m2.copy_(m2_backup); self.m3.copy_(m3_backup)
            return False
        return True

    def update_momentum(self, new_beta):
        self.beta = new_beta

    def step(self, x_in: torch.Tensor, md_verbose: bool = False, batch_idx: int = 0):
        self.step_idx += 1
        W1, W2, W3 = self.block.W1.weight, self.block.W2.weight, self.block.W3.weight
        
        # 记录初始范数
        if batch_idx % 50 == 0:
            n1_init = W1.norm().item()
            n2_init = W2.norm().item()
            n3_init = W3.norm().item()

        if self.wd != 0.0:
            with torch.no_grad():
                W1.mul_(1 - self.lr * self.wd)
                W2.mul_(1 - self.lr * self.wd)
                W3.mul_(1 - self.lr * self.wd)

        with torch.no_grad():
            self._update_ema_stats(x_in)

        # --- P1: 高压熔断逻辑 ---
        pressure = self._gauge_pressure()
        
        is_fail_safe = False
        if pressure > max(0.35, self.cfg.pressure_threshold * 3):
            is_fail_safe = True
            self._apply_gauge_fail_safe()
            
            # 覆盖径向逆映射参数
            self._md_rel_clip_override = 0.08
            self._md_dual_clip_override = 0.10
            self._md_iters_override    = 4
            self._md_tau_override      = 0.2
            
            # 短暂关闭动量
            self.momentum_off_steps = max(self.momentum_off_steps, 200)
        else:
            self._safe_apply_gauge()
            # 清除覆盖
            self._md_rel_clip_override = None
            self._md_dual_clip_override = None
            self._md_iters_override = None
            self._md_tau_override = None


        # 对偶变量 η_t
        with torch.no_grad():
            eta1, eta2, eta3 = psi_nabla_block(W1.detach(), W2.detach(), W3.detach())

        gW1, gW2, gW3 = W1.grad, W2.grad, W3.grad
        if any(x is None for x in (gW1, gW2, gW3)):
            return

        with torch.no_grad():
            # 块级梯度裁剪
            gnorm = torch.sqrt(gW1.norm()**2 + gW2.norm()**2 + gW3.norm()**2)
            if gnorm > 1.0:
                scale = 1.0 / (gnorm + 1e-12)
                gW1.mul_(scale); gW2.mul_(scale); gW3.mul_(scale)

            # 动量更新
            current_beta = self.beta
            if self.momentum_off_steps > 0:
                current_beta = 0.0
                self.momentum_off_steps -= 1
                
            self.m1.mul_(current_beta).add_(gW1)
            self.m2.mul_(current_beta).add_(gW2)
            self.m3.mul_(current_beta).add_(gW3)

            # 对偶步
            eta1_new = eta1 - self.head_lr * self.m1
            eta2_new = eta2 - self.head_lr * self.m2
            eta3_new = eta3 - self.head_lr * self.m3

        # 径向逆映射参数设置
        do_full = (self.step_idx % self.cfg.md_full_interval == 0) or (pressure > self.cfg.pressure_threshold)
        iters = self._md_iters_override if self._md_iters_override is not None else (12 if do_full else 6)
        tau_md = self._md_tau_override if self._md_tau_override is not None else 0.3
        rel_clip = self._md_rel_clip_override if self._md_rel_clip_override is not None else 0.15
        dual_clip = self._md_dual_clip_override if self._md_dual_clip_override is not None else 0.25
        
        W1_new, W2_new, W3_new = md_inverse_map_block_radial(
            W1, W2, W3, eta1_new, eta2_new, eta3_new,
            iters=iters, tau=tau_md,
            s_min=1e-4, s_max=100.0,
            alpha_floor=5e-6, eps_norm_factor=1e-6,
            rel_clip=rel_clip, dual_clip=dual_clip,
            verbose=(md_verbose and (do_full or is_fail_safe) and batch_idx < 10)
        )

        with torch.no_grad():
            W1.copy_(W1_new)
            W2.copy_(W2_new)
            W3.copy_(W3_new)
            
            # --- P2: 尺度夹紧 (防止过度收缩) ---
            if batch_idx % 50 == 0:
                gm_init = (n1_init * n2_init * n3_init + 1e-12) ** (1/3)
                n1_final = W1.norm().item()
                n2_final = W2.norm().item()
                n3_final = W3.norm().item()
                gm_final = (n1_final * n2_final * n3_final + 1e-12) ** (1/3)
                ratio = gm_final / (gm_init + 1e-12)

                floor_ratio = 2
                if ratio < floor_ratio:
                    gain = (floor_ratio / max(ratio, 1e-6)) ** (1/3)
                    W1.mul_(gain); W2.mul_(gain); W3.mul_(gain)
                    # 动量也需要反向缩放
                    self.m1.mul_(1/gain); self.m2.mul_(1/gain); self.m3.mul_(1/gain)
                    
                    n1_final = W1.norm().item()
                    n2_final = W2.norm().item()
                    n3_final = W3.norm().item()
                    print(f"[dbg] SCALE CLAMP: ratio={ratio:.3f}, gain={gain:.3f}. New ||W||=({n1_final:.2f},{n2_final:.2f},{n3_final:.2f})")

                rel1 = n1_final / (n1_init + 1e-8)
                rel2 = n2_final / (n2_init + 1e-8)
                rel3 = n3_final / (n3_init + 1e-8)
                print(f"[dbg] step={self.step_idx} batch={batch_idx} "
                      f"||W||=({n1_final:.2f},{n2_final:.2f},{n3_final:.2f}) "
                      f"rel=({rel1:.3f},{rel2:.3f},{rel3:.3f}) "
                      f"pressure={pressure:.3f}")


# ===========================
# §3. 训练与评估
# ===========================

def get_param_groups_mirror_gauge(model: ResNet18_SwiGLU_Classifier):
    backbone_params = [p for p in model.backbone.parameters()]
    classifier_params = [p for p in model.classifier.parameters()]
    swiglu_params = [model.swiglu.W1.weight, model.swiglu.W2.weight, model.swiglu.W3.weight]
    return backbone_params, classifier_params, swiglu_params


def train_model(name, trainloader, testloader, device, epochs, init_sd, lr, wd, betas, hidden,
                md_verbose=False):
    model = ResNet18_SwiGLU_Classifier(num_classes=100, hidden_features=hidden).to(device)
    if init_sd is not None:
        model.load_state_dict(init_sd)

    criterion = nn.CrossEntropyLoss()

    backbone_params, classifier_params, swiglu_params = get_param_groups_mirror_gauge(model)
    opt_adamw = torch.optim.AdamW(
        [{'params': backbone_params}, {'params': classifier_params}],
        lr=lr, weight_decay=wd, betas=betas
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt_adamw, T_max=epochs)

    controller = MirrorGaugeController(
        model.swiglu, lr=lr, weight_decay=wd, 
        beta_momentum=0.0,
        config=GaugeConfig()
    )

    train_losses, test_accs = [], []

    for ep in range(epochs):
        model.train()
        run_loss = 0.0
        start_time = time.time()
        
        # 延迟动量启动
        if ep == 6:
            controller.update_momentum(0.5)
            print(f"[Info] Epoch {ep+1}: Momentum updated to 0.5")
        elif ep == 10:
            controller.update_momentum(0.9)
            print(f"[Info] Epoch {ep+1}: Momentum updated to 0.9")
        
        for batch_idx, (x, y) in enumerate(trainloader):
            x, y = x.to(device), y.to(device)

            opt_adamw.zero_grad(set_to_none=True)
            for p in swiglu_params:
                if p.grad is not None:
                    p.grad.zero_()

            f = model.backbone(x)
            
            if not torch.isfinite(f).all():
                print(f"[WARNING] Non-finite backbone output at epoch {ep+1}, batch {batch_idx}")
                continue
                
            s = model.swiglu(f)
            
            if not torch.isfinite(s).all():
                print(f"[WARNING] Non-finite SwiGLU output at epoch {ep+1}, batch {batch_idx}")
                continue
                
            logits = model.classifier(s)
            
            if batch_idx % 50 == 0:
                with torch.no_grad():
                    logit_mean = logits.mean().item()
                    logit_std = logits.std().item()
                    if batch_idx < 200:
                        print(f"[dbg] ep={ep+1} batch={batch_idx} "
                              f"logit_mean={logit_mean:.3f} logit_std={logit_std:.3f}")
            
            if not torch.isfinite(logits).all():
                print(f"[WARNING] Non-finite logits at epoch {ep+1}, batch {batch_idx}")
                continue

            loss = criterion(logits, y)
            
            if not torch.isfinite(loss):
                print(f"[WARNING] Non-finite loss at epoch {ep+1}, batch {batch_idx}")
                continue
                
            loss.backward()

            opt_adamw.step()

            controller.step(x_in=f, md_verbose=md_verbose, batch_idx=batch_idx)

            run_loss += loss.item()

        scheduler.step()
        epoch_time = time.time() - start_time
        avg_loss = run_loss / len(trainloader)
        train_losses.append(avg_loss)

        # eval
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in testloader:
                x, y = x.to(device), y.to(device)
                f = model.backbone(x)
                s = model.swiglu(f)
                logits = model.classifier(s)
                pred = logits.argmax(1)
                total += y.size(0)
                correct += (pred == y).sum().item()
        acc = 100.0 * correct / total
        print(f"{name} | Epoch {ep+1:02d}/{epochs} | Loss={avg_loss:.4f} | Acc={acc:.2f}% | Time={epoch_time:.1f}s")
        test_accs.append(acc)

    return train_losses, test_accs, model


@torch.no_grad()
def compute_singular_values_of_swiglu_W3(model: ResNet18_SwiGLU_Classifier):
    W = model.swiglu.W3.weight.detach().cpu()
    C = W @ W.t()
    evals = torch.linalg.eigvalsh(C)
    S = torch.sqrt(torch.clamp(evals, min=0))
    S, _ = torch.sort(S, descending=True)
    return S.numpy()


# ===========================
# §4. MAIN
# ===========================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform_train = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
    ])
    transform_test = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
    ])
    
    data_dir = './data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    trainset = torchvision.datasets.CIFAR100(root=data_dir, train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root=data_dir, train=False, download=True, transform=transform_test)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

    hidden = 1024
    print("\n[Setup] Initializing base model weights ...")
    base = ResNet18_SwiGLU_Classifier(num_classes=100, hidden_features=hidden)
    init_sd = base.state_dict()
    del base
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    BASE_LR = 5e-4
    EPOCHS = 40
    WD = 0.01
    BETAS = (0.9, 0.999)
    MODEL_NAME = "Mirror-Gauge v4 (Stabilized)"

    print(f"\n===== Training with {MODEL_NAME} =====")
    print(f"[Config] head_lr_factor=0.03, s_min=1e-4, alpha_floor=5e-6, Fail-Safe enabled.")
    losses, accs, model = train_model(
        MODEL_NAME, trainloader, testloader, device, EPOCHS, init_sd,
        lr=BASE_LR, wd=WD, betas=BETAS, hidden=hidden,
        md_verbose=False
    )
    
    spectrum = compute_singular_values_of_swiglu_W3(model)
    
    os.makedirs("results", exist_ok=True)
    model_save_name = MODEL_NAME.lower().replace(' ', '_').replace('(', '').replace(')', '')
    torch.save(model.state_dict(), f"results/{model_save_name}_resnet18_swiglu_classifier.pth")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"{MODEL_NAME} on CIFAR-100 (ResNet18→SwiGLU→Linear, {EPOCHS} Epochs)", fontsize=16)

    axes[0].plot(accs, label=MODEL_NAME, marker='o', markersize=4, linewidth=2, color='C1')
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Test Accuracy (%)")
    axes[0].set_title(f"Test Accuracy (Final: {accs[-1]:.2f}%)")
    axes[0].grid(True, alpha=0.4)

    axes[1].plot(losses, label=MODEL_NAME, marker='s', markersize=4, linewidth=2, color='C0')
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Training Loss")
    axes[1].set_title("Training Loss")
    axes[1].grid(True, alpha=0.4)

    axes[2].plot(spectrum, label=MODEL_NAME, linewidth=2, color='C2')
    axes[2].set_yscale("log")
    axes[2].set_xlabel("Index")
    axes[2].set_ylabel("Singular Value")
    axes[2].set_title("Singular Spectrum of SwiGLU.W3")
    axes[2].grid(True, alpha=0.4)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_path = f"results/{model_save_name}_results.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nPlots saved to {save_path}")
    plt.show()


if __name__ == "__main__":
    main()