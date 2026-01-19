
import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from net.MultiFreqSegNetv2_lz import MultiFreqSegNetv2_lz as EnhancedQuantumLiteSeg
    print("✅ Successfully imported: EnhancedQuantumLiteSeg")
    MODEL_NAME = "EnhancedQuantumLiteSeg"
except ImportError as e:
    print(f"❌ Import failed: {e}")
    exit(1)

from dataset import SegDataset

ROOT        = "your DIR PATH/dataset"
IMG_SIZE    = 512
BATCH_SIZE  = 24
EPOCHS      = 120
LR          = 8e-4
WARMUP_EPOCHS = 5
NUM_CLASSES = 3
hidden_dim=64
CLASS_NAMES = ['Background', 'Mc mulch film', 'sub_main pipe']

SAVE_DIR    = f"./checkpoints_{MODEL_NAME}"
PLOT_DIR    = os.path.join(SAVE_DIR, "plots")
LOG_DIR     = os.path.join(SAVE_DIR, "logs")
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  Device: {DEVICE}")
print(f"🎮 Number of GPUs: {torch.cuda.device_count()}")

print("\n📊 Loading dataset...")
train_set = SegDataset(ROOT, split="train")
val_set   = SegDataset(ROOT, split="val")

print(f"Training set size: {len(train_set)}")
print(f"Validation set size: {len(val_set)}")

def analyze_class_distribution(dataset, num_samples=100):
    class_counts = [0, 0, 0]
    class_presence = [0, 0, 0]
    
    for i in range(min(num_samples, len(dataset))):
        _, mask = dataset[i]
        mask_np = mask.numpy()
        
        for j in range(3):
            if (mask_np == j).any():
                class_presence[j] += 1
            class_counts[j] += np.sum(mask_np == j)
    
    total_pixels = sum(class_counts)
    print(f"📈 Class analysis (first {num_samples} samples):")
    print(f"  Presence: Background={class_presence[0]/num_samples:.1%}, "
          f"Class1={class_presence[1]/num_samples:.1%}, "
          f"Class2 (dropper)={class_presence[2]/num_samples:.1%}")
    print(f"  Pixel ratio: Background={class_counts[0]/total_pixels:.1%}, "
          f"Class1={class_counts[1]/total_pixels:.1%}, "
          f"Class2={class_counts[2]/total_pixels:.1%}")
    
    return class_presence, class_counts

train_presence, train_counts = analyze_class_distribution(train_set, 100)
val_presence, val_counts = analyze_class_distribution(val_set, 50)

train_loader = DataLoader(
    train_set, 
    batch_size=BATCH_SIZE, 
    shuffle=True,
    num_workers=8,
    pin_memory=True,
    drop_last=False
)

val_loader = DataLoader(
    val_set,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=8,
    pin_memory=True
)

class ClassBalancedLoss(nn.Module):
    def __init__(self, num_classes=3, class_weights=None, focal_gamma=2.0):
        super().__init__()
        self.num_classes = num_classes
        
        if class_weights is None:
            self.class_weights = torch.tensor([0.5, 1.2, 3.0])
        else:
            self.class_weights = class_weights
            
        self.focal_gamma = focal_gamma
        self.epsilon = 1e-6
        
    def forward(self, pred, target):
        ce_loss = 0.0
        pred_softmax = F.softmax(pred, dim=1)
        
        for c in range(self.num_classes):
            p_c = pred_softmax[:, c, :, :]
            t_c = (target == c).float()
            focal_weight = (1 - p_c) ** self.focal_gamma
            ce_term = -t_c * torch.log(p_c + self.epsilon)
            weighted_ce = ce_term * self.class_weights[c] * focal_weight
            ce_loss += weighted_ce.mean()
        
        dice_loss = 0.0
        for c in [1, 2]:
            p_c = pred_softmax[:, c, :, :]
            t_c = (target == c).float()
            intersection = (p_c * t_c).sum()
            union = p_c.sum() + t_c.sum() + self.epsilon
            dice = 2.0 * intersection / union
            dice_loss += (1 - dice) * (3.0 if c == 2 else 1.5)
        
        total_loss = ce_loss + dice_loss * 0.5
        
        return total_loss

criterion = ClassBalancedLoss(num_classes=NUM_CLASSES, focal_gamma=2.0)

print(f"\n🧠 Initializing model: {MODEL_NAME}...")
model = EnhancedQuantumLiteSeg(num_classes=NUM_CLASSES,
    hidden_dim=hidden_dim,
    num_freq_bands=4)

model = model.to(DEVICE)
if torch.cuda.device_count() > 1:
    print(f"🚀 Using {torch.cuda.device_count()} GPUs for training")
    try:
        model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
        model = model.to(DEVICE)
    except Exception as e:
        print(f"⚠️ DataParallel failed, using single GPU: {e}")
        model = model.to(DEVICE)

def calculate_model_stats(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

total_params, trainable_params = calculate_model_stats(model)
print(f"📊 Model statistics:")
print(f"  Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
print(f"  Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")

print(f"\n⚙️  Configuring optimizer...")

quantum_params = []
classical_params = []

for name, param in model.named_parameters():
    if any(key in name for key in ['theta', 'phi', 'alpha', 'beta', 'w0', 'w1', 'w2']):
        quantum_params.append(param)
    else:
        classical_params.append(param)

optimizer = optim.AdamW([
    {'params': quantum_params, 'lr': LR * 1.5, 'weight_decay': 1e-4},
    {'params': classical_params, 'lr': LR, 'weight_decay': 1e-4}
])

scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, 
    T_max=EPOCHS,
    eta_min=LR * 0.01
)

scaler = torch.cuda.amp.GradScaler()

def compute_dice_by_class(pred, target, n_classes=3, smooth=1e-6):
    pred_labels = torch.argmax(F.softmax(pred, dim=1), dim=1)
    
    dice_list = []
    for cls in range(n_classes):
        pred_cls = (pred_labels == cls).float()
        target_cls = (target == cls).float()
        intersection = (pred_cls * target_cls).sum()
        dice = (2. * intersection + smooth) / (pred_cls.sum() + target_cls.sum() + smooth)
        dice_list.append(dice.item())
    
    mean_dice = sum(dice_list) / n_classes
    return mean_dice, dice_list

def compute_iou_by_class(pred, target, n_classes=3, smooth=1e-6):
    pred_labels = torch.argmax(F.softmax(pred, dim=1), dim=1)
    
    iou_list = []
    for cls in range(n_classes):
        pred_cls = (pred_labels == cls).float()
        target_cls = (target == cls).float()
        intersection = (pred_cls * target_cls).sum().float()
        union = pred_cls.sum().float() + target_cls.sum().float() - intersection
        iou = (intersection + smooth) / (union + smooth)
        iou_list.append(iou.item())
    
    mean_iou = sum(iou_list) / n_classes
    return mean_iou, iou_list

def compute_precision_recall_by_class(pred, target, n_classes=3, smooth=1e-6):
    pred_labels = torch.argmax(F.softmax(pred, dim=1), dim=1)
    
    precision_list = []
    recall_list = []
    
    for class_idx in range(n_classes):
        pred_cls = (pred_labels == class_idx).float()
        true_cls = (target == class_idx).float()
        tp = (pred_cls * true_cls).sum().float()
        fp = (pred_cls * (1 - true_cls)).sum().float()
        fn = ((1 - pred_cls) * true_cls).sum().float()
        precision = (tp + smooth) / (tp + fp + smooth)
        recall = (tp + smooth) / (tp + fn + smooth)
        precision_list.append(precision.item())
        recall_list.append(recall.item())
    
    mean_precision = sum(precision_list) / n_classes
    mean_recall = sum(recall_list) / n_classes
    
    return mean_precision, precision_list, mean_recall, recall_list

def validate_with_metrics(model, val_loader, device, class_names=CLASS_NAMES):
    model.eval()
    
    total_dice_scores = [0.0, 0.0, 0.0]
    total_iou_scores = [0.0, 0.0, 0.0]
    total_precision_scores = [0.0, 0.0, 0.0]
    total_recall_scores = [0.0, 0.0, 0.0]
    class_counts = [0, 0, 0]
    
    total_loss = 0.0
    
    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc="Validation"):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss.item()
            
            mean_dice, dice_scores = compute_dice_by_class(outputs, masks)
            mean_iou, iou_scores = compute_iou_by_class(outputs, masks)
            mean_precision, precision_scores, mean_recall, recall_scores = compute_precision_recall_by_class(outputs, masks)
            
            for i in range(3):
                total_dice_scores[i] += dice_scores[i]
                total_iou_scores[i] += iou_scores[i]
                total_precision_scores[i] += precision_scores[i]
                total_recall_scores[i] += recall_scores[i]
                
                if (masks == i).any():
                    class_counts[i] += 1
    
    num_batches = len(val_loader)
    avg_dice_scores = [score / num_batches for score in total_dice_scores]
    avg_iou_scores = [score / num_batches for score in total_iou_scores]
    avg_precision_scores = [score / num_batches for score in total_precision_scores]
    avg_recall_scores = [score / num_batches for score in total_recall_scores]
    
    avg_loss = total_loss / num_batches
    
    focused_dice = (avg_dice_scores[1] + avg_dice_scores[2]) / 2
    focused_iou = (avg_iou_scores[1] + avg_iou_scores[2]) / 2
    focused_precision = (avg_precision_scores[1] + avg_precision_scores[2]) / 2
    focused_recall = (avg_recall_scores[1] + avg_recall_scores[2]) / 2
    
    mean_dice = sum(avg_dice_scores) / 3
    mean_iou = sum(avg_iou_scores) / 3
    mean_precision = sum(avg_precision_scores) / 3
    mean_recall = sum(avg_recall_scores) / 3
    
    model.train()
    return {
        'val_loss': avg_loss,
        'focused_dice': focused_dice,
        'focused_iou': focused_iou,
        'focused_precision': focused_precision,
        'focused_recall': focused_recall,
        'mean_dice': mean_dice,
        'mean_iou': mean_iou,
        'mean_precision': mean_precision,
        'mean_recall': mean_recall,
        'dice_scores': avg_dice_scores,
        'iou_scores': avg_iou_scores,
        'precision_scores': avg_precision_scores,
        'recall_scores': avg_recall_scores,
        'class_counts': class_counts
    }

def train_one_epoch(model, train_loader, optimizer, device, criterion, epoch_idx):
    model.train()
    train_loss = 0.0
    
    total_class_dice = [0.0, 0.0, 0.0]
    total_class_iou = [0.0, 0.0, 0.0]
    class1_correct = 0
    class1_total = 0
    class2_correct = 0
    class2_total = 0
    
    pbar = tqdm(train_loader, desc=f"Training Epoch {epoch_idx+1}")
    for batch_idx, (images, masks) in enumerate(pbar):
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, masks)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        train_loss += loss.item()
        
        mean_dice, dice_scores = compute_dice_by_class(outputs, masks)
        mean_iou, iou_scores = compute_iou_by_class(outputs, masks)
        
        for i in range(3):
            total_class_dice[i] += dice_scores[i]
            total_class_iou[i] += iou_scores[i]
        
        pred_labels = torch.argmax(F.softmax(outputs, dim=1), dim=1)
        class1_correct += ((pred_labels == 1) & (masks == 1)).sum().item()
        class1_total += (masks == 1).sum().item()
        class2_correct += ((pred_labels == 2) & (masks == 2)).sum().item()
        class2_total += (masks == 2).sum().item()
        
        if batch_idx % 10 == 0:
            class1_acc = class1_correct / max(class1_total, 1)
            class2_acc = class2_correct / max(class2_total, 1)
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'C1_dice': f'{dice_scores[1]:.4f}',
                'C2_dice': f'{dice_scores[2]:.4f}',
                'C1_acc': f'{class1_acc:.4f}',
                'C2_acc': f'{class2_acc:.4f}'
            })
    
    avg_train_loss = train_loss / len(train_loader)
    avg_class_dice = [score / len(train_loader) for score in total_class_dice]
    avg_class_iou = [score / len(train_loader) for score in total_class_iou]
    
    class1_accuracy = class1_correct / max(class1_total, 1)
    class2_accuracy = class2_correct / max(class2_total, 1)
    
    return {
        'train_loss': avg_train_loss,
        'class_dice': avg_class_dice,
        'class_iou': avg_class_iou,
        'class1_accuracy': class1_accuracy,
        'class2_accuracy': class2_accuracy,
        'class1_total': class1_total,
        'class2_total': class2_total
    }

print(f"\n🚀 Starting training for {EPOCHS} epochs...")
print(f"📁 Models will be saved to: {SAVE_DIR}")

history = []
best_focused_dice = 0.0
best_class2_dice = 0.0
best_epoch = 0
early_stop_patience = 20
no_improve_count = 0

best_metrics = {
    'focused_dice': 0.0,
    'class2_dice': 0.0,
    'mean_iou': 0.0,
    'mean_precision': 0.0,
    'mean_recall': 0.0,
    'focused_iou': 0.0,
    'focused_precision': 0.0,
    'focused_recall': 0.0
}

start_time = time.time()

for epoch in range(EPOCHS):
    epoch_start_time = time.time()
    
    train_metrics = train_one_epoch(model, train_loader, optimizer, DEVICE, criterion, epoch)
    val_metrics = validate_with_metrics(model, val_loader, DEVICE)
    
    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']
    
    epoch_record = {
        'epoch': epoch + 1,
        'lr': current_lr,
        'train_loss': round(train_metrics['train_loss'], 4),
        'val_loss': round(val_metrics['val_loss'], 4),
        'focused_dice': round(val_metrics['focused_dice'], 5),
        'class0_dice': round(val_metrics['dice_scores'][0], 5),
        'class1_dice': round(val_metrics['dice_scores'][1], 5),
        'class2_dice': round(val_metrics['dice_scores'][2], 5),
        'train_class0_dice': round(train_metrics['class_dice'][0], 5),
        'train_class1_dice': round(train_metrics['class_dice'][1], 5),
        'train_class2_dice': round(train_metrics['class_dice'][2], 5),
        'class0_iou': round(val_metrics['iou_scores'][0], 5),
        'class1_iou': round(val_metrics['iou_scores'][1], 5),
        'class2_iou': round(val_metrics['iou_scores'][2], 5),
        'class0_precision': round(val_metrics['precision_scores'][0], 5),
        'class1_precision': round(val_metrics['precision_scores'][1], 5),
        'class2_precision': round(val_metrics['precision_scores'][2], 5),
        'class0_recall': round(val_metrics['recall_scores'][0], 5),
        'class1_recall': round(val_metrics['recall_scores'][1], 5),
        'class2_recall': round(val_metrics['recall_scores'][2], 5),
        'class1_accuracy': round(train_metrics['class1_accuracy'], 5),
        'class2_accuracy': round(train_metrics['class2_accuracy'], 5),
        'mean_iou': round(val_metrics['mean_iou'], 5),
        'mean_precision': round(val_metrics['mean_precision'], 5),
        'mean_recall': round(val_metrics['mean_recall'], 5),
        'focused_iou': round(val_metrics['focused_iou'], 5),
        'focused_precision': round(val_metrics['focused_precision'], 5),
        'focused_recall': round(val_metrics['focused_recall'], 5),
        'time': time.time() - epoch_start_time
    }
    history.append(epoch_record)
    
    print(f"\n{'='*60}")
    print(f"📊 Epoch {epoch+1:3d}/{EPOCHS} | Time: {epoch_record['time']:.1f}s")
    print(f"📈 Training loss: {epoch_record['train_loss']:.4f} | "
          f"Validation loss: {epoch_record['val_loss']:.4f} | "
          f"LR: {current_lr:.2e}")
    print(f"🎯 Focused Dice(Class1&2): {epoch_record['focused_dice']:.5f}")
    print(f"🏷️  Class Dice Val: Class0[{epoch_record['class0_dice']:.5f}] "
          f"Class1[{epoch_record['class1_dice']:.5f}] "
          f"Class2[{epoch_record['class2_dice']:.5f}]")
    print(f"🏷️  Class Dice Train: Class0[{epoch_record['train_class0_dice']:.5f}] "
          f"Class1[{epoch_record['train_class1_dice']:.5f}] "
          f"Class2[{epoch_record['train_class2_dice']:.5f}]")
    print(f"🎯 Class Accuracy: Class1[{epoch_record['class1_accuracy']:.5f}] "
          f"Class2[{epoch_record['class2_accuracy']:.5f}]")
    print(f"📐 Mean IoU: {epoch_record['mean_iou']:.5f} | "
          f"Focused IoU: {epoch_record['focused_iou']:.5f}")
    print(f"🎯 Mean Precision: {epoch_record['mean_precision']:.5f} | "
          f"Focused Precision: {epoch_record['focused_precision']:.5f}")
    print(f"📈 Mean Recall: {epoch_record['mean_recall']:.5f} | "
          f"Focused Recall: {epoch_record['focused_recall']:.5f}")
    
    if epoch_record['focused_dice'] > best_focused_dice:
        best_focused_dice = epoch_record['focused_dice']
        best_class2_dice = epoch_record['class2_dice']
        best_epoch = epoch + 1
        
        best_metrics['focused_dice'] = epoch_record['focused_dice']
        best_metrics['class2_dice'] = epoch_record['class2_dice']
        best_metrics['mean_iou'] = epoch_record['mean_iou']
        best_metrics['mean_precision'] = epoch_record['mean_precision']
        best_metrics['mean_recall'] = epoch_record['mean_recall']
        best_metrics['focused_iou'] = epoch_record['focused_iou']
        best_metrics['focused_precision'] = epoch_record['focused_precision']
        best_metrics['focused_recall'] = epoch_record['focused_recall']
        
        checkpoint_path = os.path.join(SAVE_DIR, f"best_checkpoint_epoch_{epoch+1}.pth")
        model_to_save = model.module if hasattr(model, 'module') else model
        
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_focused_dice': best_focused_dice,
            'best_class2_dice': best_class2_dice,
            'best_metrics': best_metrics,
            'config': {
                'num_classes': NUM_CLASSES,
                'img_size': IMG_SIZE,
                'model_name': MODEL_NAME
            }
        }, checkpoint_path)
        
        print(f"💾 Saved best model! Focused Dice: {best_focused_dice:.5f}, "
              f"Class2 Dice: {best_class2_dice:.5f}")
        no_improve_count = 0
    else:
        no_improve_count += 1
        print(f"⏳ No improvement in Focused Dice, streak: {no_improve_count}/{early_stop_patience}")
    
    if epoch_record['class2_dice'] > best_metrics['class2_dice']:
        best_metrics['class2_dice'] = epoch_record['class2_dice']
        model_to_save = model.module if hasattr(model, 'module') else model
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model_to_save.state_dict(),
            'class2_dice': epoch_record['class2_dice'],
            'focused_dice': epoch_record['focused_dice']
        }, os.path.join(SAVE_DIR, f"best_class2_model_dice_{epoch_record['class2_dice']:.5f}.pth"))
        print(f"💾 Saved best Class2 model! Class2 Dice: {epoch_record['class2_dice']:.5f}")
    
    if (epoch + 1) % 10 == 0:
        checkpoint_path = os.path.join(SAVE_DIR, f"checkpoint_epoch_{epoch+1}.pth")
        model_to_save = model.module if hasattr(model, 'module') else model
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_metrics': val_metrics,
            'train_metrics': train_metrics
        }, checkpoint_path)
        print(f"📂 Checkpoint saved: {checkpoint_path}")
    
    if no_improve_count >= early_stop_patience:
        print(f"\n🛑 Early stopping triggered! No improvement for {early_stop_patience} consecutive epochs")
        break
    
    print('='*60)

total_time = time.time() - start_time
print(f"\n✅ Training completed! Total time: {total_time/60:.1f} minutes")

final_model_path = os.path.join(SAVE_DIR, "final_model.pth")
model_to_save = model.module if hasattr(model, 'module') else model
torch.save({
    'model_state_dict': model_to_save.state_dict(),
    'config': {
        'num_classes': NUM_CLASSES,
        'img_size': IMG_SIZE,
        'model_name': MODEL_NAME
    },
    'best_metrics': best_metrics,
    'final_epoch': epoch + 1
}, final_model_path)
print(f"💾 Final model saved: {final_model_path}")

df = pd.DataFrame(history)
history_path = os.path.join(SAVE_DIR, "training_history.csv")
df.to_csv(history_path, index=False)
print(f"📊 Training history saved: {history_path}")

best_metrics_path = os.path.join(SAVE_DIR, "best_metrics.txt")
with open(best_metrics_path, 'w') as f:
    f.write("Best Performance Metrics Summary\n")
    f.write("=" * 50 + "\n")
    f.write(f"Model: {MODEL_NAME}\n")
    f.write(f"Best Epoch: {best_epoch}\n")
    f.write(f"Training Time: {total_time/60:.1f} minutes\n")
    f.write(f"Early Stopping: {'Yes' if no_improve_count >= early_stop_patience else 'No'}\n")
    f.write("\nBest Metrics:\n")
    f.write("=" * 50 + "\n")
    f.write(f"Focused Dice (Class1&2): {best_metrics['focused_dice']:.5f}\n")
    f.write(f"Class 2 Dice: {best_metrics['class2_dice']:.5f}\n")
    f.write(f"Mean IoU: {best_metrics['mean_iou']:.5f}\n")
    f.write(f"Focused IoU (Class1&2): {best_metrics['focused_iou']:.5f}\n")
    f.write(f"Mean Precision: {best_metrics['mean_precision']:.5f}\n")
    f.write(f"Focused Precision (Class1&2): {best_metrics['focused_precision']:.5f}\n")
    f.write(f"Mean Recall: {best_metrics['mean_recall']:.5f}\n")
    f.write(f"Focused Recall (Class1&2): {best_metrics['focused_recall']:.5f}\n")
    
    best_epoch_data = [h for h in history if h['epoch'] == best_epoch][0]
    f.write("\nClass-specific Metrics (Best Epoch):\n")
    f.write("=" * 50 + "\n")
    for class_idx in range(3):
        f.write(f"{CLASS_NAMES[class_idx]}:\n")
        f.write(f"  Val Dice: {best_epoch_data[f'class{class_idx}_dice']:.5f}\n")
        f.write(f"  Train Dice: {best_epoch_data[f'train_class{class_idx}_dice']:.5f}\n")
        f.write(f"  IoU: {best_epoch_data[f'class{class_idx}_iou']:.5f}\n")
        f.write(f"  Precision: {best_epoch_data[f'class{class_idx}_precision']:.5f}\n")
        f.write(f"  Recall: {best_epoch_data[f'class{class_idx}_recall']:.5f}\n")
        if class_idx in [1, 2]:
            f.write(f"  Train Accuracy: {best_epoch_data[f'class{class_idx}_accuracy']:.5f}\n")
        f.write("-" * 30 + "\n")

print(f"📋 Best metrics saved: {best_metrics_path}")

print(f"\n{'='*60}")
print("🎯 Training Summary")
print('='*60)
print(f"📈 Best Focused Dice(Class1&2): {best_metrics['focused_dice']:.5f} (Epoch {best_epoch})")
print(f"🏆 Best Class 2 Dice: {best_metrics['class2_dice']:.5f}")
print(f"📊 Best Dice scores per class:")
print(f"   Class0: {max([h['class0_dice'] for h in history]):.5f}")
print(f"   Class1: {max([h['class1_dice'] for h in history]):.5f}")
print(f"   Class2: {max([h['class2_dice'] for h in history]):.5f}")
print(f"\n📐 Best IoU Metrics:")
print(f"   Mean IoU: {best_metrics['mean_iou']:.5f}")
print(f"   Focused IoU(Class1&2): {best_metrics['focused_iou']:.5f}")
print(f"   Class0 IoU: {max([h['class0_iou'] for h in history]):.5f}")
print(f"   Class1 IoU: {max([h['class1_iou'] for h in history]):.5f}")
print(f"   Class2 IoU: {max([h['class2_iou'] for h in history]):.5f}")
print(f"\n🎯 Best Precision Metrics:")
print(f"   Mean Precision: {best_metrics['mean_precision']:.5f}")
print(f"   Focused Precision(Class1&2): {best_metrics['focused_precision']:.5f}")
print(f"   Class0 Precision: {max([h['class0_precision'] for h in history]):.5f}")
print(f"   Class1 Precision: {max([h['class1_precision'] for h in history]):.5f}")
print(f"   Class2 Precision: {max([h['class2_precision'] for h in history]):.5f}")
print(f"\n📈 Best Recall Metrics:")
print(f"   Mean Recall: {best_metrics['mean_recall']:.5f}")
print(f"   Focused Recall(Class1&2): {best_metrics['focused_recall']:.5f}")
print(f"   Class0 Recall: {max([h['class0_recall'] for h in history]):.5f}")
print(f"   Class1 Recall: {max([h['class1_recall'] for h in history]):.5f}")
print(f"   Class2 Recall: {max([h['class2_recall'] for h in history]):.5f}")
print(f"\n⏱️  Total training time: {total_time/60:.1f} minutes")
print(f"📁 Models saved in: {SAVE_DIR}")
print(f"📋 Metrics saved in: {best_metrics_path}")
print(f"📊 Training history saved in: {history_path}")
print('='*60)

print(f"\n✨ Training script execution completed!")
```