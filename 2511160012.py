# -*- coding: utf-8 -*-
"""
Mirror-Gauge V6 (Symplectic Nesterov with Channel-Wise Alignment)
----------------------------------------------------------------
核心改进：
1. 通道级 G31: 在 Hidden Dimension 粒度上强制执行 ||W1||~||W2||~||W3||
2. 辛动量输运: 权重 x Scale → 动量 / Scale，保持切向量方向一致
3. 纯一阶 Nesterov: 抛弃二阶矩，仅使用 Nesterov 动量 + 规范几何流
"""

import os
import math
import time
from dataclasses import dataclass
from typing import List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt


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
    def __init__(self, in_features: int, hidden_features: int, out_features: int):
        super().__init__()
        # W1, W2: [Hidden x In], W3: [Out x Hidden]
        self.W1 = nn.Linear(in_features, hidden_features, bias=False)
        self.W2 = nn.Linear(in_features, hidden_features, bias=False)
        self.W3 = nn.Linear(hidden_features, out_features, bias=False)
        
        # 初始化增强
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
        return self.classifier()


# ===========================
# §1. Mirror-Gauge V6 Controller
# ===========================

@dataclass
class GaugeConfigV6:
    # EMA 参数
    ema_decay: float = 0.9
    # 通道对齐强度 (软约束硬度)
    stiffness: float = 1  # 0.0~1.0，每次修正规范误差的比例
    # Warmup 步数
    warmup_steps: int = 200
    # 动量输运修正系数 (1.0 = 严格物理输运)
    transport_gamma: float = 1.0


class MirrorGaugeControllerV6:
    """
    Mirror-Gauge V6: 
    - 通道级 G31 规范变换
    - 辛动量输运 (Symplectic Transport)
    - 纯一阶 Nesterov 动量
    """
    def __init__(self, block: SwiGLUBlock, lr: float, weight_decay: float,
                 beta_momentum: float = 0.9, config: GaugeConfigV6 = GaugeConfigV6()):
        self.block = block
        self.lr = lr
        self.wd = weight_decay
        self.beta = beta_momentum
        self.cfg = config
        
        # Nesterov 动量缓存 (角向分量)
        self.m1 = torch.zeros_like(block.W1.weight)
        self.m2 = torch.zeros_like(block.W2.weight)
        self.m3 = torch.zeros_like(block.W3.weight)
        
        # 径向动量 (对数空间)
        self.v_rad = torch.zeros(3, device=block.W1.weight.device)
        
        self.step_idx = 0

    @torch.no_grad()
    def _apply_symplectic_channel_gauge(self):
        """
        核心：通道级 G31 规范变换 + 辛动量输运
        对每个 Hidden Unit 独立进行几何平均，并同步输运动量
        """
        W1, W2, W3 = self.block.W1.weight, self.block.W2.weight, self.block.W3.weight
        eps = 1e-10
        
        # 1. 计算通道范数 (Log Space)
        # W1/W2: [H, D_in] -> norm over dim=1 -> [H]
        # W3: [D_out, H] -> norm over dim=0 -> [H]
        n1 = torch.norm(W1, p=2, dim=1) + eps
        n2 = torch.norm(W2, p=2, dim=1) + eps
        n3 = torch.norm(W3, p=2, dim=0) + eps
        
        log_n1, log_n2, log_n3 = torch.log(n1), torch.log(n2), torch.log(n3)
        
        # 2. 计算目标 Log Norm (几何平均)
        target_log = (log_n1 + log_n2 + log_n3) / 3.0
        
        # 3. 计算修正 Scaling Factors
        # stiffness 控制修正速度，避免震荡
        d1 = (target_log - log_n1) * self.cfg.stiffness
        d2 = (target_log - log_n2) * self.cfg.stiffness
        d3 = (target_log - log_n3) * self.cfg.stiffness
        
        # 转换回线性空间
        s1 = torch.exp(d1)  # [H]
        s2 = torch.exp(d2)  # [H]
        s3 = torch.exp(d3)  # [H]
        
        # 4. 执行权重变换 (广播)
        # s1, s2 需要 unsqueeze(1) 广播到 [H, D_in]
        # s3 需要 unsqueeze(0) 广播到 [D_out, H]
        W1.mul_(s1.unsqueeze(1))
        W2.mul_(s2.unsqueeze(1))
        W3.mul_(s3.unsqueeze(0))
        
        # 5. 辛动量输运 (逆变规则: Weight * S => Momentum / S^gamma)
        def transport_momentum(momentum, scale, dim_unsq):
            inv_scale = torch.pow(scale, -self.cfg.transport_gamma)
            momentum.div_(inv_scale.unsqueeze(dim_unsq))
        
        transport_momentum(self.m1, s1, 1)
        transport_momentum(self.m2, s2, 1)
        transport_momentum(self.m3, s3, 0)
        
        # 6. 同步修正梯度 (如果存在)
        if W1.grad is not None:
            W1.grad.div_(s1.unsqueeze(1))
        if W2.grad is not None:
            W2.grad.div_(s2.unsqueeze(1))
        if W3.grad is not None:
            W3.grad.div_(s3.unsqueeze(0))
        
        return s1, s2, s3

    @torch.no_grad()
    def _apply_channel_alignment(self):
        """通道级精细对齐 (可选的附加调整)"""
        W2, W3 = self.block.W2.weight, self.block.W3.weight
        eps = 1e-8
        
        # 计算范数差异 (行 vs 列)
        w2_r = torch.norm(W2, p=2, dim=1)
        w3_c = torch.norm(W3, p=2, dim=0)
        log_ratio = torch.log(w2_r + eps) - torch.log(w3_c + eps)
        
        # 温和对齐 (学习率式的更新)
        scale = torch.exp(-self.cfg.stiffness * log_ratio)
        scale = torch.clamp(scale, 0.95, 1.05)  # 限制变化幅度
        
        W2.mul_(scale.unsqueeze(1))
        W3.mul_((1.0 / scale).unsqueeze(0))
        
        # 辛动量输运
        self.m2.div_(scale.unsqueeze(1))
        self.m3.div_((1.0 / scale).unsqueeze(0))
        if W2.grad is not None:
            W2.grad.div_(scale.unsqueeze(1))
        if W3.grad is not None:
            W3.grad.mul_(scale.unsqueeze(0))  # 注意这里是倒数

    @torch.no_grad()
    def _update_nesterov_momentum(self, W, g, m, n, rg):
        """
        纯一阶 Nesterov 动量更新 (无 v_t 二阶矩)
        公式: m_t = β * m_{t-1} + (1-β) * g_t
        更新: ΔW = -lr * (β * m_t + (1-β) * g_t)
        """
        # 切向梯度 = 梯度 - 径向投影
        g_orth = g - (rg / (n.pow(2) + 1e-12)) * W
        
        # 更新动量 (累积)
        m.mul_(self.beta).add_(g_orth, alpha=1 - self.beta)
        
        # Nesterov 前瞻向量 (Lookahead)
        # v_nest = β * m_t + (1-β) * g_t (使用当前梯度，而非历史)
        nest_vec = self.beta * m + (1 - self.beta) * g_orth
        
        # 沿切向移动
        W.add_(nest_vec, alpha=-self.lr)

    @torch.no_grad()
    def step(self, x_in: torch.Tensor, md_verbose: bool = False, batch_idx: int = 0):
        self.step_idx += 1
        W1, W2, W3 = self.block.W1.weight, self.block.W2.weight, self.block.W3.weight
        g1, g2, g3 = W1.grad, W2.grad, W3.grad
        
        if any(x is None for x in (g1, g2, g3)):
            return

        # ------------------------------------------------
        # 1. 规范预处理: 通道级 G31 + 辛动量输运
        # ------------------------------------------------
        # 在计算任何更新前，先把参数拉回规范平面
        scales = self._apply_symplectic_channel_gauge()
        
        # ------------------------------------------------
        # 2. 径向更新 (Log-Space Mirror Descent)
        # ------------------------------------------------
        n1, n2, n3 = W1.norm(), W2.norm(), W3.norm()
        
        # 点积: <W, g>
        rg1 = torch.sum(W1 * g1).item()
        rg2 = torch.sum(W2 * g2).item()
        rg3 = torch.sum(W3 * g3).item()
        
        # 对数空间梯度 (加入 Weight Decay)
        eps = 1e-12
        grad_log_s1 = (rg1 / (n1.pow(2) + eps)) + self.wd
        grad_log_s2 = (rg2 / (n2.pow(2) + eps)) + self.wd
        grad_log_s3 = (rg3 / (n3.pow(2) + eps)) + self.wd
        
        # 共享径向动量 (因为 G31 强制三者等范数)
        mean_grad_log_s = (grad_log_s1 + grad_log_s2 + grad_log_s3) / 3.0
        
        # 更新径向动量 (对数空间)
        self.v_rad.mul_(self.beta).add_(mean_grad_log_s)
        
        # 径向步长
        delta_log = -self.lr * self.v_rad.mean()
        
        # ------------------------------------------------
        # 3. 角向更新: Nesterov 动量
        # ------------------------------------------------
        self._update_nesterov_momentum(W1, g1, self.m1, n1, rg1)
        self._update_nesterov_momentum(W2, g2, self.m2, n2, rg2)
        self._update_nesterov_momentum(W3, g3, self.m3, n3, rg3)
        
        # ------------------------------------------------
        # 4. 重构: 强制径向尺度
        # ------------------------------------------------
        current_log_s = (torch.log(W1.norm() + eps) + torch.log(W2.norm() + eps) + torch.log(W3.norm() + eps)) / 3.0
        target_log_s = current_log_s + delta_log
        
        # 全局缩放 (保持 G31 结构)
        scale_step = torch.exp(delta_log)
        W1.mul_(scale_step)
        W2.mul_(scale_step)
        W3.mul_(scale_step)
        
        # 动量跟随全局缩放 (保持方向)
        self.m1.mul_(scale_step)
        self.m2.mul_(scale_step)
        self.m3.mul_(scale_step)
        
        # ------------------------------------------------
        # 5. 通道对齐 (可选的精细调整)
        # ------------------------------------------------
        if self.step_idx > self.cfg.warmup_steps:
            self._apply_channel_alignment()
        
        # Debug
        if batch_idx % 100 == 0:
            print(f"[MG-V6] step={self.step_idx:04d} | "
                  f"||W||=({W1.norm():6.2f}, {W2.norm():6.2f}, {W3.norm():6.2f}) | "
                  f"RadVel={self.v_rad.mean():+.2e} | "
                  f"ChScales=({scales[0].mean():.3f}, {scales[1].mean():.3f}, {scales[2].mean():.3f})")


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

    controller = MirrorGaugeControllerV6(
        model.swiglu, lr=lr, weight_decay=wd, 
        beta_momentum=0.9,
        config=GaugeConfigV6(stiffness=0.2, warmup_steps=200)
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
            logits = model.classifier()
            
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
                logits = model.classifier()
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
    MODEL_NAME = "Mirror-Gauge V6 (Symplectic Nesterov)"

    print(f"\n===== Training with {MODEL_NAME} =====")
    print(f"[Config] Channel-wise G31, Symplectic Transport, Pure 1st-Order Nesterov")
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
