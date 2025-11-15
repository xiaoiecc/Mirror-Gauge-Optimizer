# -*- coding: utf-8 -*-
"""
Mirror-Gauge V5 (Log-Space Stable)
模型结构：ResNet-18 → SwiGLU → Linear
核心改进：对数空间动力学 + 严格规范约束
- 彻底消除塌缩问题 (V4 的黑洞缺陷)
- 无需 floor_ratio 人工呼吸
- G31 作为流形回缩，绝对稳定
- 径向与角向动力学解耦
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
# §0. 模型组件 (保持不变)
# ===========================

class ResNet18Backbone(nn.Module):
    def __init__(self, pretrained: bool = False):
        super().__init__()
        model = torchvision.models.resnet18(weights=None if not pretrained else torchvision.models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Identity()
        self.backbone = model
        self.out_features = 512

    def forward(self, x):
        return self.backbone(x)


class SwiGLUBlock(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, out_features: int):
        super().__init__()
        self.W1 = nn.Linear(in_features, hidden_features, bias=False)
        self.W2 = nn.Linear(in_features, hidden_features, bias=False)
        self.W3 = nn.Linear(hidden_features, out_features, bias=False)
        nn.init.kaiming_uniform_(self.W1.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W2.weight, a=math.sqrt(5))
        nn.init.xavier_uniform_(self.W3.weight)

    def forward(self, x):
        a = self.W1(x)
        b = self.W2(x)
        h = F.silu(a) * b
        return self.W3(h)


class ResNet18_SwiGLU_Classifier(nn.Module):
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
# §1. Mirror-Gauge V5 Controller
# ===========================

@dataclass
class GaugeConfig:
    # EMA 参数 (不变)
    ema_decay: float = 0.9
    # 通道对齐强度 (可调节，默认保守值)
    eta_g32: float = 0.05
    clip_channel_scale: float = 0.1
    # Warmup (可选，V5 更稳定)
    warmup_steps: int = 100


class MirrorGaugeController:
    """
    Mirror-Gauge V5: Log-Space Stable Dynamics
    核心：在对数空间处理径向更新，几何平均执行 G31，彻底避免塌缩
    """
    def __init__(self, block: SwiGLUBlock, lr: float, weight_decay: float,
                 beta_momentum: float = 0.9, config: GaugeConfig = GaugeConfig()):
        self.block = block
        self.lr = lr
        self.wd = weight_decay
        self.beta = beta_momentum
        self.cfg = config
        
        # 动量缓存 (角度分量)
        self.m1 = torch.zeros_like(block.W1.weight)
        self.m2 = torch.zeros_like(block.W2.weight)
        self.m3 = torch.zeros_like(block.W3.weight)
        
        # 径向动量 (对数空间)
        self.v_rad = torch.zeros(3, device=block.W1.weight.device)
        
        # EMA 统计量
        H = block.W1.weight.shape[0]
        device = block.W1.weight.device
        self.ema_coef = torch.zeros(H, device=device)
        self.ema_zeta = torch.zeros(H, device=device)
        self.ema_ratio = torch.zeros(H, device=device)
        self.ema_silu_prime = torch.ones(H, device=device) * 0.5
        
        self.step_idx = 0

    @torch.no_grad()
    def _update_ema_stats(self, x_in: torch.Tensor):
        """更新 SwiGLU 非线性行为的 EMA 统计"""
        W1, W2, W3 = self.block.W1.weight, self.block.W2.weight, self.block.W3.weight
        a = x_in @ W1.t()
        b = x_in @ W2.t()
        sigma = torch.sigmoid(a)
        silu_prime = sigma * (1 + a * (1 - sigma))
        silu_a = F.silu(a)

        coef = (1.0 - silu_prime).mean(dim=0)
        num = (a * silu_prime).mean(dim=0)
        den = silu_a.mean(dim=0) + 1e-8
        zeta = num / den
        ratio = (b / (a.abs() + 1e-5)).mean(dim=0)
        silu_p = silu_prime.mean(dim=0)

        d = self.cfg.ema_decay
        self.ema_coef.mul_(d).add_((1 - d) * coef)
        self.ema_zeta.mul_(d).add_((1 - d) * zeta)
        self.ema_ratio.mul_(d).add_((1 - d) * ratio)
        self.ema_silu_prime.mul_(d).add_((1 - d) * silu_p)

    @torch.no_grad()
    def _apply_gauge_G31_log_space(self):
        """
        V5 核心: 在 Log 空间执行严格 G31 规范变换
        强制 ln(s1) = ln(s2) = ln(s3)，实现几何平均
        这是流形上的 Retraction，绝对稳定
        """
        W1, W2, W3 = self.block.W1.weight, self.block.W2.weight, self.block.W3.weight
        eps = 1e-12
        
        # 计算对数范数
        log_s1 = torch.log(W1.norm() + eps)
        log_s2 = torch.log(W2.norm() + eps)
        log_s3 = torch.log(W3.norm() + eps)
        
        # 目标对数范数 (几何平均)
        target_log_s = (log_s1 + log_s2 + log_s3) / 3.0
        
        # 计算修正因子
        d1 = target_log_s - log_s1
        d2 = target_log_s - log_s2
        d3 = target_log_s - log_s3
        
        # 执行乘法修正 (log space delta => exp(delta) scaling)
        W1.mul_(torch.exp(d1))
        W2.mul_(torch.exp(d2))
        W3.mul_(torch.exp(d3))
        
        # 动量跟随规范变换，保持一致性
        self.m1.mul_(torch.exp(d1))
        self.m2.mul_(torch.exp(d2))
        self.m3.mul_(torch.exp(d3))

    @torch.no_grad()
    def _apply_channel_alignment(self):
        """通道级规范变换：对齐 W2 行范数与 W3 列范数"""
        W2, W3 = self.block.W2.weight, self.block.W3.weight
        eps = 1e-8
        
        # 计算范数差异
        w2_r = torch.norm(W2, p=2, dim=1)
        w3_c = torch.norm(W3, p=2, dim=0)
        log_ratio = torch.log(w2_r + eps) - torch.log(w3_c + eps)
        
        # 温和对齐 (学习率式的更新)
        scale = torch.exp(-self.cfg.eta_g32 * log_ratio)
        scale = torch.clamp(scale, 1 - self.cfg.clip_channel_scale, 1 + self.cfg.clip_channel_scale)
        
        W2.mul_(scale.unsqueeze(1))
        W3.mul_((1.0 / scale).unsqueeze(0))
        
        self.m2.mul_(scale.unsqueeze(1))
        self.m3.mul_((1.0 / scale).unsqueeze(0))

    @torch.no_grad()
    def step(self, x_in: torch.Tensor, md_verbose: bool = False, batch_idx: int = 0):
        self.step_idx += 1
        W1, W2, W3 = self.block.W1.weight, self.block.W2.weight, self.block.W3.weight
        g1, g2, g3 = W1.grad, W2.grad, W3.grad
        
        if any(x is None for x in (g1, g2, g3)):
            return

        # ------------------------------------------------
        # 1. 规范预处理: G31 强制几何平均
        # ------------------------------------------------
        # 在计算任何更新前，先把参数拉回规范平面
        self._apply_gauge_G31_log_space()
        
        # 更新 EMA 统计 (用于通道对齐)
        self._update_ema_stats(x_in)
        
        # ------------------------------------------------
        # 2. 计算径向梯度 (Log-Space 梯度)
        # ------------------------------------------------
        n1 = W1.norm()
        n2 = W2.norm()
        n3 = W3.norm()
        
        # 点积: <W, g>
        rg1 = torch.sum(W1 * g1).item()
        rg2 = torch.sum(W2 * g2).item()
        rg3 = torch.sum(W3 * g3).item()
        
        # 对数空间梯度 (加入 Weight Decay)
        eps = 1e-12
        grad_log_s1 = (rg1 / (n1.pow(2) + eps)) + self.wd
        grad_log_s2 = (rg2 / (n2.pow(2) + eps)) + self.wd
        grad_log_s3 = (rg3 / (n3.pow(2) + eps)) + self.wd
        
        # ------------------------------------------------
        # 3. 径向动量更新 (Log-Space Mirror Descent)
        # ------------------------------------------------
        # 由于 G31 强制三者等范数，我们可以共享径向动量
        mean_grad_log_s = (grad_log_s1 + grad_log_s2 + grad_log_s3) / 3.0
        
        # 更新径向动量 (对数空间)
        radial_lr = self.lr * 1.0
        self.v_rad[0] = self.beta * self.v_rad[0] + mean_grad_log_s
        self.v_rad[1] = self.beta * self.v_rad[1] + mean_grad_log_s
        self.v_rad[2] = self.beta * self.v_rad[2] + mean_grad_log_s
        
        # 计算对数尺度变化量
        delta_log_s1 = -radial_lr * self.v_rad[0]
        delta_log_s2 = -radial_lr * self.v_rad[1]
        delta_log_s3 = -radial_lr * self.v_rad[2]
        
        # ------------------------------------------------
        # 4. 角向更新 (切空间动量)
        # ------------------------------------------------
        def apply_orthogonal_update(W, g, m, n, rg):
            # 切向梯度 = 梯度 - 径向投影
            g_orth = g - (rg / (n.pow(2) + 1e-12)) * W
            
            # 更新动量
            m.mul_(self.beta).add_(g_orth)
            
            # 沿切向移动 (这会轻微改变模长)
            W.add_(m, alpha=-self.lr)
        
        apply_orthogonal_update(W1, g1, self.m1, n1, rg1)
        apply_orthogonal_update(W2, g2, self.m2, n2, rg2)
        apply_orthogonal_update(W3, g3, self.m3, n3, rg3)
        
        # ------------------------------------------------
        # 5. 重构: 强制径向尺度
        # ------------------------------------------------
        # 当前的模长 (经过角向移动后)
        current_log_s1 = torch.log(W1.norm() + 1e-12)
        current_log_s2 = torch.log(W2.norm() + 1e-12)
        current_log_s3 = torch.log(W3.norm() + 1e-12)
        
        # 目标模长 (G31 后的平均 + 径向更新)
        prev_log_s = (torch.log(n1 + 1e-12) + torch.log(n2 + 1e-12) + torch.log(n3 + 1e-12)) / 3.0
        target_log_s1 = prev_log_s + delta_log_s1
        target_log_s2 = prev_log_s + delta_log_s2
        target_log_s3 = prev_log_s + delta_log_s3
        
        # 强制重置到目标尺度
        W1.mul_(torch.exp(target_log_s1 - current_log_s1))
        W2.mul_(torch.exp(target_log_s2 - current_log_s2))
        W3.mul_(torch.exp(target_log_s3 - current_log_s3))
        
        # ------------------------------------------------
        # 6. 通道对齐 (可选的精细调整)
        # ------------------------------------------------
        if self.step_idx > self.cfg.warmup_steps:
            self._apply_channel_alignment()
        
        # 打印调试信息
        if batch_idx % 100 == 0:
            print(f"[Mirror-Gauge V5] step={self.step_idx:04d} | "
                  f"||W||=({W1.norm():6.2f}, {W2.norm():6.2f}, {W3.norm():6.2f}) | "
                  f"LogGrad={mean_grad_log_s:+.2e}")


# ===========================
# §2. 训练与评估
# ===========================

def get_param_groups_mirror_gauge(model: ResNet18_SwiGLU_Classifier):
    backbone_params = [p for p in model.backbone.parameters()]
    classifier_params = [p for p in model.classifier.parameters()]
    swiglu_params = [model.swiglu.W1.weight, model.swiglu.W2.weight, model.swiglu.W3.weight]
    return backbone_params, classifier_params, swiglu_params


def train_model(name, trainloader, testloader, device, epochs, init_sd, lr, wd, betas, hidden, md_verbose=False):
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
        beta_momentum=0.9,
        config=GaugeConfig()
    )

    train_losses, test_accs = [], []

    for ep in range(epochs):
        model.train()
        run_loss = 0.0
        start_time = time.time()
        
        for batch_idx, (x, y) in enumerate(trainloader):
            x, y = x.to(device), y.to(device)

            opt_adamw.zero_grad(set_to_none=True)
            for p in swiglu_params:
                if p.grad is not None:
                    p.grad.zero_()

            f = model.backbone(x)
            s = model.swiglu(f)
            logits = model.classifier(s)
            
            loss = criterion(logits, y)
            loss.backward()

            opt_adamw.step()
            controller.step(x_in=f, md_verbose=md_verbose, batch_idx=batch_idx)

            run_loss += loss.item()

        scheduler.step()
        epoch_time = time.time() - start_time
        avg_loss = run_loss / len(trainloader)
        train_losses.append(avg_loss)

        # 评估
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
# §3. MAIN
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
    MODEL_NAME = "Mirror-Gauge V5 (Log-Space Stable)"

    print(f"\n===== Training with {MODEL_NAME} =====")
    print(f"[Config] 无 floor_ratio, 无 Fail-Safe")
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
