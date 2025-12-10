import argparse
import os
import sys
import time
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from datasets.mavceleb_encoded import MAVCelebEncodedDataset
from models.crossmodal_verification import CrossModalVerificationModel


class OrthogonalProjectionLoss(nn.Module):
    """
    Contrastive loss that pulls same-class embeddings together
    and pushes different-class embeddings apart
    """
    def __init__(self, device):
        super(OrthogonalProjectionLoss, self).__init__()
        self.device = device

    def forward(self, features, labels):
        """
        Args:
            features: (batch_size, embed_dim) - normalized embeddings
            labels: (batch_size,) - person IDs
        
        Returns:
            loss: scalar contrastive loss
        """
        # Normalize features
        features = F.normalize(features, p=2, dim=1)

        labels = labels[:, None]

        # Create masks for positive and negative pairs
        mask = torch.eq(labels, labels.t()).bool().to(self.device)
        eye = torch.eye(mask.shape[0], mask.shape[1]).bool().to(self.device)

        mask_pos = mask.masked_fill(eye, 0).float()
        mask_neg = (~mask).float()
        
        # Compute dot products (similarity matrix)
        dot_prod = torch.matmul(features, features.t())

        # Average similarity for positive and negative pairs
        pos_pairs_mean = (mask_pos * dot_prod).sum() / (mask_pos.sum() + 1e-6)
        neg_pairs_mean = torch.abs(mask_neg * dot_prod).sum() / (mask_neg.sum() + 1e-6)

        # Loss: maximize positive similarity, minimize negative similarity
        loss = (1.0 - pos_pairs_mean) + (0.7 * neg_pairs_mean)

        return loss, pos_pairs_mean, neg_pairs_mean


def get_args_parser():
    parser = argparse.ArgumentParser('Cross-modal Speaker Verification Training', add_help=False)
    
    # Data parameters
    parser.add_argument('--data_file', type=str, required=True,
                        help='Path to training data file')
    parser.add_argument('--encoded_root', type=str, required=True,
                        help='Root directory of pre-encoded features')
    parser.add_argument('--data_root', type=str, 
                        default='/data/user_data/zeyangz/MAV-Celeb_v3',
                        help='Original data root')
    
    # Model parameters
    parser.add_argument('--audio_dim', type=int, default=1024,
                        help='Audio feature dimension')
    parser.add_argument('--image_dim', type=int, default=1536,
                        help='Image feature dimension')
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='Hidden dimension')
    parser.add_argument('--embed_dim', type=int, default=256,
                        help='Embedding dimension')
    parser.add_argument('--num_classes', type=int, default=50,
                        help='Number of person identities')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate')
    parser.add_argument('--activation', type=str, default='gelu',
                        choices=['relu', 'gelu'], help='Activation function')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size per GPU')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-6,
                        help='Minimum learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='Weight decay')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='Warmup epochs')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='Weight for contrastive loss (OPL)')
    
    # System parameters
    parser.add_argument('--num_workers', type=int, default=12,
                        help='Number of data loading workers')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--resume', type=str, default='',
                        help='Resume from checkpoint')
    parser.add_argument('--save_freq', type=int, default=5,
                        help='Save checkpoint every N epochs')
    
    # Distributed parameters
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--dist_url', type=str, default='env://')
    
    return parser


def setup_distributed():
    """Initialize distributed training"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        print('Not using distributed mode')
        return False, 0, 1, 0
    
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    dist.barrier()
    
    return True, rank, world_size, local_rank


def cosine_scheduler(base_value, final_value, epochs, warmup_epochs=0):
    """Cosine learning rate scheduler with warmup"""
    warmup_schedule = np.linspace(0, base_value, warmup_epochs)
    
    iters = np.arange(epochs - warmup_epochs)
    schedule = final_value + 0.5 * (base_value - final_value) * \
               (1 + np.cos(np.pi * iters / len(iters)))
    
    schedule = np.concatenate((warmup_schedule, schedule))
    return schedule


def train_one_epoch(model, dataloader, criterion, opl_criterion, optimizer, epoch, args, device, log_file):
    """Train for one epoch with both classification and contrastive loss"""
    model.train()
    
    total_loss = 0.0
    total_ce_loss = 0.0
    total_opl_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (audio, image, labels) in enumerate(dataloader):
        audio = audio.to(device)
        image = image.to(device)
        labels = labels.to(device)
        
        # Forward pass
        logits, audio_embed, image_embed, fused_features = model(audio, image)
        
        # Compute classification loss
        loss_ce = criterion(logits, labels)
        
        # Compute contrastive loss (OrthogonalProjectionLoss)
        loss_opl, pos_sim, neg_sim = opl_criterion(fused_features, labels)
        
        # Combined loss
        loss = loss_ce + args.alpha * loss_opl
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        total_ce_loss += loss_ce.item()
        total_opl_loss += loss_opl.item()
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        if batch_idx % 10 == 0 and args.local_rank == 0:
            print(f'Epoch [{epoch}] Batch [{batch_idx}/{len(dataloader)}] '
                  f'Loss: {loss.item():.4f} (CE: {loss_ce.item():.4f}, OPL: {loss_opl.item():.4f}) '
                  f'Acc: {100.*correct/total:.2f}%')
    
    avg_loss = total_loss / len(dataloader)
    avg_ce_loss = total_ce_loss / len(dataloader)
    avg_opl_loss = total_opl_loss / len(dataloader)
    accuracy = 100. * correct / total
    
    # Log to file
    if args.local_rank == 0 and log_file:
        import json
        log_entry = {
            'epoch': epoch,
            'train_loss': avg_loss,
            'train_loss_ce': avg_ce_loss,
            'train_loss_opl': avg_opl_loss,
            'train_accuracy': accuracy,
            'train_lr': optimizer.param_groups[0]['lr'],
            'alpha': args.alpha,
            'n_parameters': sum(p.numel() for p in model.parameters())
        }
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    return avg_loss, accuracy


def save_checkpoint(model, optimizer, epoch, args, filename='checkpoint.pth'):
    """Save checkpoint"""
    if args.local_rank == 0:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'args': args
        }
        filepath = os.path.join(args.output_dir, filename)
        torch.save(checkpoint, filepath)
        print(f'Checkpoint saved to {filepath}')


def main(args):
    # Setup distributed training
    is_distributed, rank, world_size, local_rank = setup_distributed()
    args.local_rank = local_rank
    device = torch.device(f'cuda:{local_rank}')
    
    if rank == 0:
        print("=" * 80)
        print("Cross-modal Speaker Verification Training")
        print("=" * 80)
        print(f"Data file: {args.data_file}")
        print(f"Encoded features: {args.encoded_root}")
        print(f"Number of classes: {args.num_classes}")
        print(f"Batch size per GPU: {args.batch_size}")
        print(f"Total batch size: {args.batch_size * world_size}")
        print(f"Epochs: {args.epochs}")
        print(f"Learning rate: {args.lr}")
        print(f"Activation: {args.activation}")
        print("=" * 80)
    
    # Create dataset
    dataset = MAVCelebEncodedDataset(
        data_file=args.data_file,
        encoded_root=args.encoded_root,
        data_root=args.data_root,
        is_train=True
    )
    
    if rank == 0:
        print(f"Dataset size: {len(dataset)}")
    
    # Create dataloader
    if is_distributed:
        sampler = DistributedSampler(dataset, shuffle=True)
    else:
        sampler = None
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    # Create model
    model = CrossModalVerificationModel(
        audio_input_dim=args.audio_dim,
        image_input_dim=args.image_dim,
        hidden_dim=args.hidden_dim,
        embed_dim=args.embed_dim,
        num_classes=args.num_classes,
        dropout=args.dropout,
        activation=args.activation
    )
    
    model = model.to(device)
    
    if is_distributed:
        model = DDP(model, device_ids=[local_rank])
    
    # Criteria: Classification + Contrastive
    criterion = nn.CrossEntropyLoss()
    opl_criterion = OrthogonalProjectionLoss(device=device)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    lr_schedule = cosine_scheduler(
        base_value=args.lr,
        final_value=args.min_lr,
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            if rank == 0:
                print(f"Resumed from checkpoint: {args.resume} (epoch {checkpoint['epoch']})")
    
    # Create log file
    log_file = os.path.join(args.output_dir, 'log.txt') if rank == 0 else None
    
    # Training loop
    if rank == 0:
        print("\nStarting training...")
    
    for epoch in range(start_epoch, args.epochs):
        if is_distributed:
            sampler.set_epoch(epoch)
        
        # Update learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_schedule[epoch]
        
        if rank == 0:
            print(f"\n{'=' * 80}")
            print(f"Epoch {epoch}/{args.epochs} - LR: {lr_schedule[epoch]:.6f}")
            print(f"{'=' * 80}")
        
        # Train one epoch
        avg_loss, accuracy = train_one_epoch(
            model, dataloader, criterion, opl_criterion, optimizer, epoch, args, device, log_file
        )
        
        if rank == 0:
            print(f"Epoch {epoch} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        
        # Save checkpoint
        if rank == 0 and (epoch + 1) % args.save_freq == 0:
            save_checkpoint(model, optimizer, epoch, args, f'checkpoint_epoch_{epoch}.pth')
    
    # Save final checkpoint
    if rank == 0:
        save_checkpoint(model, optimizer, args.epochs - 1, args, 'checkpoint_final.pth')
        print("\nTraining completed!")
    
    if is_distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Cross-modal verification training', parents=[get_args_parser()])
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    main(args)

