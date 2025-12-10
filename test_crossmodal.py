import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets.mavceleb_encoded import MAVCelebEncodedTestDataset
from models.crossmodal_verification import CrossModalVerificationModel


def compute_scores(model, dataloader, device):
    """
    Compute distance scores between audio and image embeddings
    Lower score = more similar (likely same person)
    """
    model.eval()
    
    all_audio_embeds = []
    all_image_embeds = []
    all_keys = []
    
    print("Extracting embeddings...")
    with torch.no_grad():
        for batch_idx, (audio, image, _, keys) in enumerate(dataloader):
            audio = audio.to(device)
            image = image.to(device)
            
            # Get embeddings (ignore fused_features and logits for testing)
            _, audio_embed, image_embed, _ = model(audio, image)
            
            all_audio_embeds.append(audio_embed.cpu().numpy())
            all_image_embeds.append(image_embed.cpu().numpy())
            all_keys.extend(keys)
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {(batch_idx + 1) * len(audio)} samples...")
    
    # Concatenate all embeddings
    audio_embeds = np.concatenate(all_audio_embeds, axis=0)
    image_embeds = np.concatenate(all_image_embeds, axis=0)
    
    print(f"Total samples: {len(audio_embeds)}")
    print(f"Audio embedding shape: {audio_embeds.shape}")
    print(f"Image embedding shape: {image_embeds.shape}")
    
    # Compute L2 distances
    print("Computing L2 distances...")
    scores = np.linalg.norm(audio_embeds - image_embeds, axis=1)
    
    return scores, all_keys


def save_scores(scores, keys, output_file):
    """Save scores to file"""
    with open(output_file, 'w') as f:
        for key, score in zip(keys, scores):
            f.write(f'{key} {score:.6f}\n')
    print(f"Scores saved to {output_file}")


def evaluate_test_set(model, test_file, encoded_root, device, output_file, args, test_lang):
    """Evaluate on a test set"""
    print(f"\nEvaluating {test_lang}: {test_file}")
    print(f"Output: {output_file}")
    
    # Create dataset
    dataset = MAVCelebEncodedTestDataset(
        data_file=test_file,
        encoded_root=encoded_root,
        data_root=args.data_root
    )
    
    print(f"Test dataset size: {len(dataset)}")
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Compute scores
    scores, keys = compute_scores(model, dataloader, device)
    
    # Save scores
    save_scores(scores, keys, output_file)
    
    # Print statistics
    print(f"\nScore statistics:")
    print(f"  Mean: {scores.mean():.4f}")
    print(f"  Std:  {scores.std():.4f}")
    print(f"  Min:  {scores.min():.4f}")
    print(f"  Max:  {scores.max():.4f}")


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 80)
    print("Cross-modal Speaker Verification Testing")
    print("=" * 80)
    print(f"Training language: {args.train_lang}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Encoded features: {args.encoded_root}")
    print(f"Device: {device}")
    print("=" * 80)
    
    # Load model
    print("\nLoading model...")
    model = CrossModalVerificationModel(
        audio_input_dim=args.audio_dim,
        image_input_dim=args.image_dim,
        hidden_dim=args.hidden_dim,
        embed_dim=args.embed_dim,
        num_classes=args.num_classes,
        dropout=args.dropout,
        activation=args.activation
    )
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Remove 'module.' prefix if exists (from DDP)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict)
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # Determine heard/unheard languages
    train_lang = args.train_lang
    
    # Evaluate on English test set
    if args.english_test:
        # English is "heard" if trained on English, "unheard" if trained on German
        status = 'heard' if train_lang == 'English' else 'unheard'
        output_file = os.path.join(args.output_dir, f'sub_score_English_{status}.txt')
        
        evaluate_test_set(
            model=model,
            test_file=args.english_test,
            encoded_root=args.encoded_root,
            device=device,
            output_file=output_file,
            args=args,
            test_lang='English'
        )
    
    # Evaluate on German test set
    if args.german_test:
        # German is "heard" if trained on German, "unheard" if trained on English
        status = 'heard' if train_lang == 'German' else 'unheard'
        output_file = os.path.join(args.output_dir, f'sub_score_German_{status}.txt')
        
        evaluate_test_set(
            model=model,
            test_file=args.german_test,
            encoded_root=args.encoded_root,
            device=device,
            output_file=output_file,
            args=args,
            test_lang='German'
        )
    
    print("\n" + "=" * 80)
    print("Testing completed!")
    print("=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Cross-modal verification testing')
    
    # Data parameters
    parser.add_argument('--train_lang', type=str, required=True,
                        choices=['English', 'German'],
                        help='Language used for training (to determine heard/unheard)')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--encoded_root', type=str, required=True,
                        help='Root directory of pre-encoded features')
    parser.add_argument('--data_root', type=str,
                        default='/data/user_data/zeyangz/MAV-Celeb_v3',
                        help='Original data root')
    parser.add_argument('--english_test', type=str, default='',
                        help='English test file')
    parser.add_argument('--german_test', type=str, default='',
                        help='German test file')
    
    # Model parameters
    parser.add_argument('--audio_dim', type=int, default=1024)
    parser.add_argument('--image_dim', type=int, default=1536)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--num_classes', type=int, default=50)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--activation', type=str, default='gelu')
    
    # System parameters
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=12)
    parser.add_argument('--output_dir', type=str, required=True)
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    main(args)

