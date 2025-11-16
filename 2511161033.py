# -*- coding: utf-8 -*-
"""
Mirror-Gauge V7: Analytic Projective Dual (解析射影对偶版)
---------------------------------------------------------
CIFAR-100 Experiment
Model: ResNet18 Backbone + SwiGLU Block + Linear Head

核心特性:
1. 解析射影逆映射 (Analytic Projective Inverse Map): 
   利用熵势能下的 Bregman 投影等价于几何平均的性质，
   一步解析算出满足 G31 约束的最优尺度，无需迭代。
2. 对偶空间动量 (Dual Momentum):
   在对数空间 (Log-Space) 累积尺度动量，实现乘性更新。
3. 精细几何整形 (Fine-Grained Geometry):
   保留 V4 的 G21/G22/G32 逻辑，利用 EMA 统计量微调 SiLU 的非线性形状。
"""

import os
import math
import time
from dataclasses import dataclass
from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np


# ===========================
# §0. 模型定义 (ResNet18 + SwiGLU)
# ===========================

class ResNet18Backbone(nn.Module):
    def __init__(self, pretrained: bool = False):
        super().__init__()
        weights = torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
        model = torchvision.models.resnet18(weights=weights)
        model.fc = nn.Identity()
        self.backbone = model
        self.out_features = 512

    def forward(self, x):
        return self.backbone(x)


class SwiGLUBlock(nn.Module):
    """
    SwiGLU(x) = (SiLU(xW1^T) ⊙ (xW2^T)) W3^T
    Linear weights shape: [Out, In]
    W1: [Hidden, In]
    W2: [Hidden, In]
    W3: [Out, Hidden]
    """
    def __init__(self, in_features: int, hidden_features: int, out_features: int):
        super().__init__()
        self.W1 = nn.Linear(in_features, hidden_features, bias=False)
        self.W2 = nn.Linear(in_features, hidden_features, bias=False)
        self.W3 = nn.Linear(hidden_features, out_features, bias=False)
        
        # 初始化
        nn.init.kaiming_uniform_(self.W1.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W2.weight, a=math.sqrt(5))
        nn.init.xavier_uniform_(self.W3.weight)

    def forward(self, x):
        a = self.W1(x)
        b = self.W2(x)
        h = F.silu(a) * b
        return self.W3(h)


class ResNet18_SwiGLU_Classifier(nn.Module):
    def __init__(self, num_classes: int = 100, hidden_features: int = 1024):
        super().__init__()
        self.backbone = ResNet18Backbone()
        feat_dim = self.backbone.out_features
        self.swiglu = SwiGLUBlock(feat_dim, hidden_features, feat_dim)
        self.classifier = nn.Linear(feat_dim, num_classes, bias=True)

    def forward(self, x):
        f = self.backbone(x)
        s = self.swiglu(f)
        return self.classifier(s)


# ===========================
# §1. Mirror-Gauge V7 Controller
# ===========================

@dataclass
class GaugeConfigV7:
    # 动力学参数
    lr: float = 1e-3
    beta_momentum: float = 0.9
    weight_decay: float = 0.01
    
    # V4 几何参数 (用于精细微调投影偏置)
    eta_g21: float = 0.05   # SiLU Linearization
    eta_g22: float = 0.03   # Channel Compensation
    eta_g32: float = 0.05   # W2-W3 Coupling (Output-Input)
    
    # 统计平滑与预热
    ema_decay: float = 0.9
    warmup_steps: int = 200
    
    # 投影硬度 (1.0 = 强制 G31, <1.0 = 允许部分松弛)
    projection_hardness: float = 1.0 


class MirrorGaugeControllerV7:
    """
    Mirror-Gauge V7: Analytic Projective Dual
    ------------------------------------------
    核心流程:
    1. Primal -> Dual: 计算当前 Log Norm (eta)
    2. Dual Update: 在对偶空间应用自然梯度和动量 (eta_new = eta - lr * m_dual)
    3. Inverse Map: 使用"解析射影"将 eta_new 映射回满足规范约束的流形 (Geometric Mean + Shape Bias)
    4. Directional Update: 在切空间更新方向 (Orthogonal Nesterov)
    5. Reconstruction: W = Direction * exp(Projected_Dual)
    """
    def __init__(self, block: SwiGLUBlock, config: GaugeConfigV7):
        self.block = block
        self.cfg = config
        self.step_idx = 0
        self.eps = 1e-10
        
        # --- 1. 对偶空间动量 (Log-Space Momentum) ---
        # 对应通道维度的 Log Scale 变化率
        H = block.W1.weight.shape[0]
        device = block.W1.weight.device
        
        self.m_dual_1 = torch.zeros(H, device=device)
        self.m_dual_2 = torch.zeros(H, device=device)
        self.m_dual_3 = torch.zeros(H, device=device)
        
        # --- 2. 角向动量 (Directional Momentum) ---
        # 对应切空间的方向变化
        self.m_dir_1 = torch.zeros_like(block.W1.weight)
        self.m_dir_2 = torch.zeros_like(block.W2.weight)
        self.m_dir_3 = torch.zeros_like(block.W3.weight)
        
        # --- 3. EMA 统计量 (用于几何微调) ---
        self.ema_coef = torch.zeros(H, device=device)
        self.ema_zeta = torch.zeros(H, device=device)

    @torch.no_grad()
    def _update_ema_stats(self, x_in: torch.Tensor):
        """计算 SiLU 的非线性统计特征 (复用 V4 逻辑)"""
        W1 = self.block.W1.weight
        # x_in: [B, In], W1: [H, In] -> a: [B, H]
        a = x_in @ W1.t()
        sigma = torch.sigmoid(a)
        silu_prime = sigma * (1 + a*(1 - sigma))
        
        # 统计量 1: 线性度系数
        coef = (1.0 - silu_prime).mean(dim=0)
        
        # 统计量 2: 门控非线性补偿
        den = F.silu(a).mean(dim=0).abs() + 1e-6
        zeta = (a * silu_prime).mean(dim=0) / den
        
        d = self.cfg.ema_decay
        self.ema_coef.mul_(d).add_((1-d) * coef)
        self.ema_zeta.mul_(d).add_((1-d) * zeta)

    @torch.no_grad()
    def _projective_inverse_map(self, eta1, eta2, eta3):
        """
        【解析射影逆映射】
        将三个独立的对偶状态 (Log Scales) 投影到 SwiGLU 的最优规范流形上。
        基准流形: eta1 = eta2 = eta3 (几何平均)
        形状修正: 由 EMA 统计量决定的 Bias
        """
        # 1. 几何平均中心
        mean_eta = (eta1 + eta2 + eta3) / 3.0
        
        # 2. 计算形状偏置 (Shape Bias)
        # 限制幅度防止震荡
        clip_val = 0.5
        
        # G21: SiLU 线性化补偿 (W1 需要稍大)
        bias_1 = self.cfg.eta_g21 * torch.clamp(self.ema_coef, -clip_val, clip_val)
        
        # G22: 通道补偿 (W2 稍小, W3 稍大)
        zeta_clamped = torch.clamp(self.ema_zeta, -clip_val, clip_val)
        bias_2 = -self.cfg.eta_g22 * zeta_clamped
        bias_3 = +self.cfg.eta_g22 * zeta_clamped
        
        # 3. 调整偏置中心以保持总能量守恒
        # 我们希望 sum(final_eta) ≈ sum(input_eta)
        bias_center = (bias_1 + bias_2 + bias_3) / 3.0
        bias_1 -= bias_center
        bias_2 -= bias_center
        bias_3 -= bias_center
        
        # 4. 计算目标状态
        target_1 = mean_eta + bias_1
        target_2 = mean_eta + bias_2
        target_3 = mean_eta + bias_3
        
        # 5. 软投影 (Projection Hardness)
        h = self.cfg.projection_hardness
        final_1 = (1-h) * eta1 + h * target_1
        final_2 = (1-h) * eta2 + h * target_2
        final_3 = (1-h) * eta3 + h * target_3
        
        # 返回逆映射结果 (Exp)
        return torch.exp(final_1), torch.exp(final_2), torch.exp(final_3)

    @torch.no_grad()
    def step(self, x_in: torch.Tensor, md_verbose: bool = False, batch_idx: int = 0):
        self.step_idx += 1
        W1, W2, W3 = self.block.W1.weight, self.block.W2.weight, self.block.W3.weight
        g1, g2, g3 = W1.grad, W2.grad, W3.grad
        
        if any(x is None for x in (g1, g2, g3)): return

        # --- Phase 0: 统计更新 ---
        self._update_ema_stats(x_in)
        
        # --- Phase 1: 对偶空间更新 (The Scale) ---
        
        # 1.1 映射到对偶空间 (Log Norm)
        # W1, W2: [H, In] -> norm dim=1 -> [H]
        # W3: [Out, H] -> norm dim=0 -> [H]
        n1 = W1.norm(dim=1) + self.eps
        n2 = W2.norm(dim=1) + self.eps
        n3 = W3.norm(dim=0) + self.eps
        
        eta1_curr = torch.log(n1)
        eta2_curr = torch.log(n2)
        eta3_curr = torch.log(n3)
        
        # 1.2 计算对偶梯度 (Natural Gradient for Scale)
        # g_dual = <g, W> (per channel)
        g_dual_1 = torch.sum(W1 * g1, dim=1)
        g_dual_2 = torch.sum(W2 * g2, dim=1)
        g_dual_3 = torch.sum(W3 * g3, dim=0)
        
        # 添加 Weight Decay (在对偶空间的表现形式)
        # L2 Regularization R(W) = 0.5 * wd * ||W||^2
        # dR/d(log n) = wd * n^2
        wd_term_1 = self.cfg.weight_decay * n1.pow(2)
        wd_term_2 = self.cfg.weight_decay * n2.pow(2)
        wd_term_3 = self.cfg.weight_decay * n3.pow(2)
        
        # 1.3 更新对偶动量
        self.m_dual_1.mul_(self.cfg.beta_momentum).add_(g_dual_1 + wd_term_1)
        self.m_dual_2.mul_(self.cfg.beta_momentum).add_(g_dual_2 + wd_term_2)
        self.m_dual_3.mul_(self.cfg.beta_momentum).add_(g_dual_3 + wd_term_3)
        
        # 1.4 试探性对偶更新 (Tentative Step)
        # 使用 Riemannian 步长归一化: divide by n^2
        # step = lr * momentum / n^2
        eta1_next = eta1_curr - self.cfg.lr * (self.m_dual_1 / (n1.pow(2) + 1e-6))
        eta2_next = eta2_curr - self.cfg.lr * (self.m_dual_2 / (n2.pow(2) + 1e-6))
        eta3_next = eta3_curr - self.cfg.lr * (self.m_dual_3 / (n3.pow(2) + 1e-6))
        
        # 1.5 解析射影逆映射 (The Inverse Map)
        # 这是 V7 的核心：强制将独立的尺度拉回规范流形
        s1_new, s2_new, s3_new = self._projective_inverse_map(eta1_next, eta2_next, eta3_next)
        
        # --- Phase 2: 切空间更新 (The Direction) ---
        
        def update_direction(W, g, m, n_curr, dim_norm):
            # 2.1 计算切向梯度 (Orthogonal Gradient)
            # g_rad = <W, g> / ||W||^2 * W
            if dim_norm == 1: # Row-wise (W1, W2)
                dot = torch.sum(W * g, dim=1, keepdim=True)
                w_norm_sq = n_curr.pow(2).unsqueeze(1)
                g_orth = g - (dot / (w_norm_sq + 1e-10)) * W
            else: # Col-wise (W3)
                dot = torch.sum(W * g, dim=0, keepdim=True)
                w_norm_sq = n_curr.pow(2).unsqueeze(0)
                g_orth = g - (dot / (w_norm_sq + 1e-10)) * W
            
            # 2.2 更新角向动量 (Standard Nesterov)
            m.mul_(self.cfg.beta_momentum).add_(g_orth)
            
            # 2.3 更新方向
            # 黎曼步长: lr / ||W||
            # 为了广播正确，需要 unsqueeze
            if dim_norm == 1:
                scale = n_curr.unsqueeze(1)
            else:
                scale = n_curr.unsqueeze(0)
            
            scale_factor = torch.max(scale, torch.tensor(1.0, device=W.device))
            W.sub_(m * (self.cfg.lr / scale_factor))
            
            # 2.4 强制归一化方向 (Project back to Sphere)
            # 因为我们把尺度和方向解耦了，这里只需保留方向信息
            if dim_norm == 1:
                W.div_(W.norm(dim=1, keepdim=True) + 1e-10)
            else:
                W.div_(W.norm(dim=0, keepdim=True) + 1e-10)

        update_direction(W1, g1, self.m_dir_1, n1, dim_norm=1)
        update_direction(W2, g2, self.m_dir_2, n2, dim_norm=1)
        update_direction(W3, g3, self.m_dir_3, n3, dim_norm=0)
        
        # --- Phase 3: 重构 (Reconstruction) ---
        # W_final = Direction * Scale
        W1.mul_(s1_new.unsqueeze(1))
        W2.mul_(s2_new.unsqueeze(1))
        W3.mul_(s3_new.unsqueeze(0))
        
        # Debug Info
        if md_verbose and batch_idx % 100 == 0:
            print(f"[MG-V7] Step {self.step_idx} | "
                  f"||W||=({n1.mean():.2f}, {n2.mean():.2f}, {n3.mean():.2f}) | "
                  f"Log-Dual-Mom=({self.m_dual_1.mean():.2e})")


# ===========================
# §2. 训练与评估工具
# ===========================

def get_param_groups(model: ResNet18_SwiGLU_Classifier):
    # 分离 SwiGLU 参数和其他参数
    swiglu_params = [model.swiglu.W1.weight, model.swiglu.W2.weight, model.swiglu.W3.weight]
    swiglu_ids = list(map(id, swiglu_params))
    
    other_params = [p for p in model.parameters() if id(p) not in swiglu_ids]
    
    return other_params, swiglu_params

def train_cifar100(model_name, epochs=40, lr=1e-3, wd=0.01, batch_size=128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 数据准备
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
    os.makedirs(data_dir, exist_ok=True)
    trainset = torchvision.datasets.CIFAR100(root=data_dir, train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root=data_dir, train=False, download=True, transform=transform_test)
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # 模型初始化
    model = ResNet18_SwiGLU_Classifier(num_classes=100, hidden_features=1024).to(device)
    
    # 优化器设置
    other_params, swiglu_params = get_param_groups(model)
    
    # 1. Backbone & Head 使用 AdamW
    opt_adamw = torch.optim.AdamW(other_params, lr=lr, weight_decay=wd)
    
    # 2. SwiGLU 使用 Mirror-Gauge V7
    # 注意：Mirror-Gauge 不需要 PyTorch 优化器包装 SwiGLU 参数，它手动更新
    # 我们只需要确保 AdamW 不更新 SwiGLU 参数 (通过上面分离 params 实现)
    
    # Mirror-Gauge Config
    mg_config = GaugeConfigV7(
        lr=lr, 
        weight_decay=wd,
        beta_momentum=0.9,
        projection_hardness=1.0 # 强规范约束
    )
    controller = MirrorGaugeControllerV7(model.swiglu, mg_config)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt_adamw, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    # 记录
    train_losses = []
    test_accs = []
    
    print(f"\n===== Training {model_name} (Mirror-Gauge V7) =====")
    
    for ep in range(epochs):
        model.train()
        run_loss = 0.0
        start_time = time.time()
        
        for batch_idx, (x, y) in enumerate(trainloader):
            x, y = x.to(device), y.to(device)

            # Zero Grad
            opt_adamw.zero_grad(set_to_none=True)
            for p in swiglu_params:
                if p.grad is not None: p.grad.zero_()

            # Forward
            # 我们需要手动拆分 forward 才能把 input 传给 controller
            # 为了兼容代码结构，直接调用 model(x) 也可以，
            # 但我们需要 SwiGLU 的输入特征用于统计 EMA
            # 所以这里手动 forward
            f = model.backbone(x)
            s = model.swiglu(f)
            logits = model.classifier(s)
            
            loss = criterion(logits, y)
            loss.backward()

            # Optim Steps
            opt_adamw.step() # Update Backbone & Head
            
            # Mirror-Gauge Update SwiGLU
            controller.step(x_in=f, md_verbose=(batch_idx==0), batch_idx=batch_idx)

            run_loss += loss.item()

        # Scheduler Step (AdamW LR decay)
        # 注意: Mirror-Gauge 的 LR 目前是固定的，也可以手动 decay
        scheduler.step()
        
        # Optional: Decay MG LR same as AdamW
        current_lr = scheduler.get_last_lr()[0]
        controller.cfg.lr = current_lr

        epoch_time = time.time() - start_time
        avg_loss = run_loss / len(trainloader)
        train_losses.append(avg_loss)

        # Eval
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in testloader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                pred = logits.argmax(1)
                total += y.size(0)
                correct += (pred == y).sum().item()
        acc = 100.0 * correct / total
        print(f"Epoch {ep+1:02d}/{epochs} | Loss={avg_loss:.4f} | Acc={acc:.2f}% | Time={epoch_time:.1f}s")
        test_accs.append(acc)

    return train_losses, test_accs, model


# ===========================
# §3. MAIN
# ===========================

def main():
    losses, accs, model = train_cifar100(
        model_name="MG-V7-ResNet18-SwiGLU", 
        epochs=40, # 可根据需要调整
        lr=1e-3, 
        wd=0.01
    )
    
    # Plotting
    os.makedirs("results", exist_ok=True)
    
    # Compute Spectrum of W3
    W3 = model.swiglu.W3.weight.detach().cpu()
    S = torch.linalg.svdvals(W3)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].plot(accs, marker='o', color='C1')
    axes[0].set_title(f"Test Acc (Final: {accs[-1]:.2f}%)")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(losses, marker='s', color='C0')
    axes[1].set_title("Train Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(S.numpy(), color='C2')
    axes[2].set_yscale('log')
    axes[2].set_title("Singular Values of W3")
    axes[2].grid(True, alpha=0.3)
    
    plt.suptitle("Mirror-Gauge V7 (Analytic Projective Dual) on CIFAR-100")
    plt.tight_layout()
    plt.savefig("results/mg_v7_cifar100.png")
    print("Results saved to results/mg_v7_cifar100.png")
    plt.show()

if __name__ == "__main__":
    main()
