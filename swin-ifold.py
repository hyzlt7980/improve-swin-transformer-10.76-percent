import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, DistributedSampler
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
import random
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp.autocast_mode import autocast
import csv
import time

# Check for timm
try:
    from timm.layers import (
        ClassifierHead, DropPath, LayerNorm, Mlp, Attention,
        create_conv2d, make_divisible, to_2tuple, trunc_normal_
    )
    from timm.models.swin_transformer import SwinTransformerBlock as SwinBlock
except ImportError:
    print("Error: timm not installed. Please install via 'pip install timm'")
    exit(1)

# =================================================================================
# SECTION 1: CORE ATTENTION MODULES
# =================================================================================

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Conv1d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv1d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x_p = x.permute(0, 2, 1)
        avg_out = self.fc(self.avg_pool(x_p))
        max_out = self.fc(self.max_pool(x_p))
        return x * self.sigmoid(avg_out + max_out).permute(0, 2, 1)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x_p = x.permute(0, 3, 1, 2)
        avg_out = torch.mean(x_p, dim=1, keepdim=True)
        max_out, _ = torch.max(x_p, dim=1, keepdim=True)
        y = self.conv1(torch.cat([avg_out, max_out], dim=1))
        return x * self.sigmoid(y).permute(0, 2, 3, 1)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)
    def forward(self, x, H, W):
        B, N, C = x.shape
        x = self.ca(x)
        x_s = x.view(B, H, W, C)
        x_s = self.sa(x_s)
        return x_s.view(B, N, C)

# =================================================================================
# SECTION 2: DISRUPTION & DENOISING BLOCKS
# =================================================================================
class EnhancedDisruptBlock(nn.Module): #swinblock+patch merging+swinblock+swinblock+swinblock

#swinblock+EnhancedDisruptBlock+patch merging+swinblock+EnhancedDisruptBlock+swinblock+hancedDisruptBlock+swin+denoiseblock

    def __init__(self, dim, input_resolution, num_heads=8, disturb_methods=['vector_amp', 'freq_mask', 'spatial_drop'], disturb_intensity=0.3, **kwargs):
        super().__init__()
        self.dim, self.disturb_methods, self.disturb_intensity = dim, disturb_methods, disturb_intensity
    
    def _disturb_features(self, x):
        B, C, H, W = x.shape
        disturbed = x.clone()
        if len(self.disturb_methods) > 0:
            selected = random.sample(self.disturb_methods, random.randint(1, len(self.disturb_methods)))
        else:
            selected = []
            
        for method in selected:
            intensity = self.disturb_intensity * (0.7 + 0.6 * torch.rand(1).item())
            
            if method == 'vector_amp':
                if W > 0:
                    cols = random.sample(range(W), max(1, int(W * intensity)))
                    disturbed[:, :, :, cols] *= (1 + 4 * torch.rand(B, 1, 1, len(cols), device=x.device))
            
            elif method == 'freq_mask':
                orig_dtype = disturbed.dtype
                x_f32 = disturbed.to(torch.float32)
                x_fft = torch.fft.rfft2(x_f32, norm="ortho")
                mask = torch.ones_like(x_fft)
                h_max, w_max = x_fft.shape[2], x_fft.shape[3]
                h_s = random.randint(0, max(0, h_max - 1))
                w_s = random.randint(0, max(0, w_max - 1))
                h_len, w_len = int(h_max * intensity), int(w_max * intensity)
                mask[:, :, h_s:h_s+h_len, w_s:w_s+w_len] = 0
                disturbed = torch.fft.irfft2(x_fft * mask, s=(H, W), norm="ortho").to(orig_dtype)
            
            elif method == 'spatial_drop':
                disturbed *= (torch.rand(B, 1, H, W, device=x.device) > intensity).float()
            
            elif method == 'channel_shuffle':
                disturbed = disturbed[:, torch.randperm(C, device=x.device), :, :]
            
            elif method == 'feature_noise':
                disturbed += torch.randn_like(x) * (intensity * torch.std(x, dim=[1,2,3], keepdim=True))
        return (0.8 * disturbed + 0.2 * x)

    def forward(self, x):
        return self._disturb_features(x) if self.training else x
class DenoisingDisruptionBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., drop_path=0., noise_level=0.3):
        super().__init__()
        self.norm1, self.attn = LayerNorm(dim), Attention(dim, num_heads=num_heads)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2, self.mlp = LayerNorm(dim), Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio))
        self.noise_level, self.denoise_loss_fn, self.denoise_proj = noise_level, nn.MSELoss(), nn.Linear(dim, dim)

    def forward(self, x, current_epoch=None):
        x_clean = x
        if self.training and (current_epoch is None or current_epoch >= 30):
            x_noisy = x + torch.randn_like(x) * self.noise_level
        else:
            x_noisy = x 
        x_p = x_noisy + self.drop_path(self.attn(self.norm1(x_noisy)))
        x_p = x_p + self.drop_path(self.mlp(self.norm2(x_p)))
        if self.training: return x_p, self.denoise_loss_fn(self.denoise_proj(x_p), x_clean.detach())
        return x_p

# =================================================================================
# SECTION 3: ASTROFORMER V4 MODEL
# =================================================================================

class SwinCBAMBlock(nn.Module):
    def __init__(self, dim, input_res, num_heads, window_size, shift_size, drop_path=0.):
        super().__init__()
        self.swin = SwinBlock(
            dim=dim, input_resolution=input_res, num_heads=num_heads,
            window_size=to_2tuple(window_size), shift_size=shift_size, drop_path=drop_path
        )
        self.cbam, self.res = CBAM(dim), to_2tuple(input_res)

    def forward(self, x):
        x = self.swin(x)
        B, H, W, C = x.shape
        return self.cbam(x.view(B, -1, C), H, W).view(B, H, W, C)

class AstroStage(nn.Module):
    def __init__(self, in_c, out_c, stride, depth, feat_size, num_heads, dpr, config):
        super().__init__()
        self.down = nn.Sequential(create_conv2d(in_c, out_c, 3, stride=stride, padding=1), nn.BatchNorm2d(out_c), nn.GELU())
        res = (feat_size[0] // stride, feat_size[1] // stride)
        self.blocks = nn.ModuleList([SwinCBAMBlock(out_c, res, num_heads, 4, 0 if i%2==0 else 2, dpr[i]) for i in range(depth)])
       
        self.disrupt = EnhancedDisruptBlock(out_c, res, num_heads, **config)

    def forward(self, x, current_epoch=None):
        x = self.down(x).permute(0, 2, 3, 1)
        for b in self.blocks: x = b(x)
        x = x.permute(0, 3, 1, 2)
        if current_epoch is not None and current_epoch < 30:
            return x
        return self.disrupt(x) if self.training else x

class AstroformerV4(nn.Module):
    def __init__(self, img_size=64, num_classes=100, embed_dims=(96, 192, 384, 768), depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24)):
        super().__init__()
        self.stem = create_conv2d(3, embed_dims[0], 3, stride=2, padding=1)
        feat_size, dpr = (img_size // 2, img_size // 2), [x.item() for x in torch.linspace(0, 0.1, sum(depths))]
        configs = [{'methods':['spatial_drop','feature_noise'],'intensity':0.02}, {'methods':['freq_mask','channel_shuffle'],'intensity':0.03}, {'methods':['vector_amp','spatial_drop'],'intensity':0.04}]
        self.stages = nn.ModuleList()
        in_c = embed_dims[0]
        for i in range(3):
            self.stages.append(AstroStage(in_c, embed_dims[i], 1 if i==0 else 2, depths[i], feat_size, num_heads[i], dpr[sum(depths[:i]):sum(depths[:i+1])], configs[i]))
            in_c, stride_f = embed_dims[i], 1 if i==0 else 2
            feat_size = (feat_size[0]//stride_f, feat_size[1]//stride_f)
        self.final_down = create_conv2d(embed_dims[2], embed_dims[3], 3, stride=2, padding=1)
        self.final = DenoisingDisruptionBlock(embed_dims[3], num_heads[3], drop_path=dpr[-depths[3]:][0])
        self.norm, self.head = LayerNorm(embed_dims[3]), nn.Linear(embed_dims[3], num_classes)

    def forward(self, x, current_epoch=None):
        x = self.stem(x)
        for s in self.stages: x = s(x, current_epoch)
        x = self.final_down(x)
        out = self.final(x.flatten(2).transpose(1, 2), current_epoch)
        feat, loss = out if isinstance(out, tuple) else (out, None)
        logits = self.head(self.norm(feat).mean(dim=1))
        return (logits, loss) if self.training and loss is not None else logits

# =================================================================================
# SECTION 4: TRAINING PIPELINE
# =================================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--warmup_epochs', type=int, default=40)
    parser.add_argument('--lr', type=float, default=8e-5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--use_denoising_loss', action='store_true')
    args = parser.parse_args()

    dist.init_process_group(backend="nccl")
    rank, world_size, local_rank = int(os.environ['RANK']), int(os.environ['WORLD_SIZE']), int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank); device = torch.device(f"cuda:{local_rank}")

    transform_train = transforms.Compose([transforms.Resize(64), 
    
    transforms.RandomCrop(64, padding=4), 
    transforms.RandomHorizontalFlip(), 
    transforms.TrivialAugmentWide(), 
    transforms.ToTensor(), 
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])

    transform_val = transforms.Compose([transforms.Resize(64), transforms.ToTensor(), transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])

    train_set = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    val_set = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_val)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=DistributedSampler(train_set, shuffle=True), num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = AstroformerV4().to(device)
    model_ddp = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
    
    optimizer = optim.AdamW(model_ddp.parameters(), lr=args.lr * world_size, weight_decay=0.05)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup_epochs)
    criterion, best_acc, start_epoch = nn.CrossEntropyLoss(), 0.0, 0

    log_file = "training_log.csv"
    if rank == 0:
        if not os.path.exists(log_file):
            with open(log_file, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Epoch', 'Loss', 'LR', 'Val_Acc', 'Best_Acc'])

    ckpt_path = './last_checkpoint.pth'
    best_weights_path = './best_linear_model.pth'
    
    if os.path.exists(ckpt_path):
        if rank == 0: print(f"==> Resuming from LAST Checkpoint (Latest Progress)...")
        checkpoint = torch.load(ckpt_path, map_location=device)
        model_ddp.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['best_acc']
    elif os.path.exists(best_weights_path):
        if rank == 0: 
            print(f"==> ⚠️ 'last_checkpoint.pth' not found.")
            print(f"==> Resuming from BEST Saved Model (Epoch 17). Progress reset to safe state.")
        checkpoint = torch.load(best_weights_path, map_location=device)
        model_ddp.module.load_state_dict(checkpoint) 
        start_epoch = 17 
        best_acc = 39.18

    # ==========================
    # Main Training Loop
    # ==========================
    for epoch in range(start_epoch, args.epochs):
        model_ddp.train(); train_loader.sampler.set_epoch(epoch)
        epoch_loss = 0.0
        start_time = time.time()
        
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            if epoch < args.warmup_epochs:
                curr_lr = args.lr * world_size * ((epoch * len(train_loader) + i) / (args.warmup_epochs * len(train_loader)))
                for pg in optimizer.param_groups: pg['lr'] = curr_lr
            
            with autocast(device_type='cuda', dtype=torch.bfloat16):
                out = model_ddp(inputs, epoch)
                logits, d_loss = out if isinstance(out, tuple) else (out, None)
                loss = criterion(logits, targets)
                if args.use_denoising_loss and d_loss is not None: loss += 0.1 * d_loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_ddp.parameters(), max_norm=0.5)
            optimizer.step()
            epoch_loss += loss.item()
            
            if rank == 0 and i % 100 == 0:
                print(f"[Epoch {epoch+1}] Step {i}/{len(train_loader)} | Loss: {loss.item():.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

        epoch_time = time.time() - start_time
        epoch_loss /= len(train_loader)

        if epoch >= args.warmup_epochs: scheduler.step()

        model_ddp.eval(); correct, total = 0, 0
        with torch.no_grad():
            # 🚀 修正：这里不再用 t，而用 targets，变量名终于对上了！
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                res = model_ddp(inputs, epoch)
                if isinstance(res, tuple): res = res[0]
                _, p = res.max(1); total += targets.size(0); correct += p.eq(targets).sum().item()
        
        metrics = torch.tensor([correct, total], dtype=torch.float32, device=device)
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        val_acc = 100. * metrics[0].item() / metrics[1].item()
        
        if rank == 0:
            print(f"==> Epoch {epoch+1} Finished ({epoch_time:.0f}s) | Avg Loss: {epoch_loss:.4f} | Validation Acc: {val_acc:.2f}%")
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model_ddp.module.state_dict(), './best_linear_model.pth')
                print(f"==> 🏆 New Best Model Saved!")
            
            with open(log_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch + 1, f"{epoch_loss:.4f}", f"{optimizer.param_groups[0]['lr']:.2e}", f"{val_acc:.2f}", f"{best_acc:.2f}"])

            torch.save({
                'epoch': epoch, 'best_acc': best_acc,
                'model_state_dict': model_ddp.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, ckpt_path)

    dist.destroy_process_group()

if __name__ == '__main__': main()
