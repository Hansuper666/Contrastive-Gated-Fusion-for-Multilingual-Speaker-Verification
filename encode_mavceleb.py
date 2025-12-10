import os
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import torchaudio
from transformers import AutoImageProcessor, AutoModel, Wav2Vec2FeatureExtractor
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
from tqdm import tqdm
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')


class ImageDataset(Dataset):
    def __init__(self, image_paths, output_paths):
        self.image_paths = image_paths
        self.output_paths = output_paths
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        return self.image_paths[idx], self.output_paths[idx]


class AudioDataset(Dataset):
    def __init__(self, audio_paths, output_paths):
        self.audio_paths = audio_paths
        self.output_paths = output_paths
    
    def __len__(self):
        return len(self.audio_paths)
    
    def __getitem__(self, idx):
        return self.audio_paths[idx], self.output_paths[idx]


def encode_images_on_gpu(gpu_id, image_paths, output_paths, model_path, progress_queue):
    """Encode images using DINOv2 on specified GPU"""
    device = torch.device(f'cuda:{gpu_id}')
    
    # Load model
    print(f"[GPU {gpu_id}] Loading DINOv2 model...")
    processor = AutoImageProcessor.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path).to(device)
    model.eval()
    
    print(f"[GPU {gpu_id}] Processing {len(image_paths)} images")
    
    with torch.no_grad():
        for img_path, out_path in zip(image_paths, output_paths):
            try:
                # Load image
                image = Image.open(img_path).convert('RGB')
                
                # Preprocess
                inputs = processor(images=image, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Extract features
                outputs = model(**inputs)
                features = outputs.last_hidden_state.cpu().numpy()
                
                # Save features
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                np.save(out_path, features)
                
                # Update progress
                if progress_queue is not None:
                    progress_queue.put(1)
                    
            except Exception as e:
                print(f"[GPU {gpu_id}] Error processing {img_path}: {e}")
    
    print(f"[GPU {gpu_id}] Finished processing images")


def encode_audio_on_gpu(gpu_id, audio_paths, output_paths, model_path, progress_queue):
    """Encode audio using WavLM on specified GPU"""
    device = torch.device(f'cuda:{gpu_id}')
    
    # Load model
    print(f"[GPU {gpu_id}] Loading WavLM model...")
    processor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path).to(device)
    model.eval()
    
    print(f"[GPU {gpu_id}] Processing {len(audio_paths)} audio files")
    
    with torch.no_grad():
        for audio_path, out_path in zip(audio_paths, output_paths):
            try:
                # Load audio
                waveform, sample_rate = torchaudio.load(audio_path)
                
                # Resample if needed
                if sample_rate != 16000:
                    resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                    waveform = resampler(waveform)
                
                # Convert to mono if stereo
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                
                # Preprocess
                inputs = processor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Extract features
                outputs = model(**inputs)
                features = outputs.last_hidden_state.cpu().numpy()
                
                # Save features
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                np.save(out_path, features)
                
                # Update progress
                if progress_queue is not None:
                    progress_queue.put(1)
                    
            except Exception as e:
                print(f"[GPU {gpu_id}] Error processing {audio_path}: {e}")
    
    print(f"[GPU {gpu_id}] Finished processing audio")


def collect_files(data_root, output_root):
    """Collect all image and audio files from the dataset"""
    image_files = []
    audio_files = []
    
    # Training data
    faces_dir = os.path.join(data_root, 'faces')
    voices_dir = os.path.join(data_root, 'voices')
    
    if os.path.exists(faces_dir):
        print("Collecting training face images...")
        for root, dirs, files in os.walk(faces_dir):
            for file in files:
                if file.endswith('.jpg'):
                    img_path = os.path.join(root, file)
                    rel_path = os.path.relpath(img_path, faces_dir)
                    out_path = os.path.join(output_root, 'faces', rel_path.replace('.jpg', '.npy'))
                    image_files.append((img_path, out_path))
    
    if os.path.exists(voices_dir):
        print("Collecting training voice files...")
        for root, dirs, files in os.walk(voices_dir):
            for file in files:
                if file.endswith('.wav'):
                    audio_path = os.path.join(root, file)
                    rel_path = os.path.relpath(audio_path, voices_dir)
                    out_path = os.path.join(output_root, 'voices', rel_path.replace('.wav', '.npy'))
                    audio_files.append((audio_path, out_path))
    
    # Test data - English
    english_test = os.path.join(data_root, 'English_test')
    if os.path.exists(english_test):
        print("Collecting English test data...")
        # Images
        face_dir = os.path.join(english_test, 'face')
        if os.path.exists(face_dir):
            for file in os.listdir(face_dir):
                if file.endswith('.jpg'):
                    img_path = os.path.join(face_dir, file)
                    out_path = os.path.join(output_root, 'English_test', 'face', file.replace('.jpg', '.npy'))
                    image_files.append((img_path, out_path))
        
        # Audio
        voice_dir = os.path.join(english_test, 'voice')
        if os.path.exists(voice_dir):
            for file in os.listdir(voice_dir):
                if file.endswith('.wav'):
                    audio_path = os.path.join(voice_dir, file)
                    out_path = os.path.join(output_root, 'English_test', 'voice', file.replace('.wav', '.npy'))
                    audio_files.append((audio_path, out_path))
    
    # Test data - German
    german_test = os.path.join(data_root, 'German_test')
    if os.path.exists(german_test):
        print("Collecting German test data...")
        # Images
        face_dir = os.path.join(german_test, 'face')
        if os.path.exists(face_dir):
            for file in os.listdir(face_dir):
                if file.endswith('.jpg'):
                    img_path = os.path.join(face_dir, file)
                    out_path = os.path.join(output_root, 'German_test', 'face', file.replace('.jpg', '.npy'))
                    image_files.append((img_path, out_path))
        
        # Audio
        voice_dir = os.path.join(german_test, 'voice')
        if os.path.exists(voice_dir):
            for file in os.listdir(voice_dir):
                if file.endswith('.wav'):
                    audio_path = os.path.join(voice_dir, file)
                    out_path = os.path.join(output_root, 'German_test', 'voice', file.replace('.wav', '.npy'))
                    audio_files.append((audio_path, out_path))
    
    return image_files, audio_files


def split_for_gpus(files, num_gpus):
    """Split files evenly across GPUs"""
    splits = [[] for _ in range(num_gpus)]
    for idx, file_pair in enumerate(files):
        splits[idx % num_gpus].append(file_pair)
    return splits


def progress_monitor(queue, total):
    """Monitor progress from all workers"""
    pbar = tqdm(total=total, desc="Processing")
    count = 0
    while count < total:
        try:
            queue.get(timeout=1)
            count += 1
            pbar.update(1)
        except:
            pass
    pbar.close()


def main():
    parser = argparse.ArgumentParser(description='Encode MAV-Celeb dataset')
    parser.add_argument('--data_root', type=str, default='/data/user_data/zeyangz/MAV-Celeb_v3',
                        help='Root directory of MAV-Celeb dataset')
    parser.add_argument('--output_root', type=str, default='/data/user_data/zeyangz/MAV-Celeb_v3/Pre-encoded',
                        help='Output directory for encoded features')
    parser.add_argument('--dinov2_path', type=str, 
                        default='/data/user_data/zeyangz/HiCMAE_LSMA/saved/model/pretrained/dinov2-giant',
                        help='Path to DINOv2 model')
    parser.add_argument('--wavlm_path', type=str,
                        default='/data/user_data/zeyangz/HiCMAE_LSMA/saved/model/pretrained/wavlm-large',
                        help='Path to WavLM model')
    parser.add_argument('--num_gpus', type=int, default=2, help='Number of GPUs to use')
    parser.add_argument('--num_workers', type=int, default=12, help='Number of workers')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("MAV-Celeb Dataset Encoding")
    print("=" * 80)
    print(f"Data root: {args.data_root}")
    print(f"Output root: {args.output_root}")
    print(f"DINOv2 model: {args.dinov2_path}")
    print(f"WavLM model: {args.wavlm_path}")
    print(f"Number of GPUs: {args.num_gpus}")
    print(f"Number of workers: {args.num_workers}")
    print("=" * 80)
    
    # Collect all files
    print("\nCollecting files...")
    image_files, audio_files = collect_files(args.data_root, args.output_root)
    
    print(f"\nFound {len(image_files)} images")
    print(f"Found {len(audio_files)} audio files")
    print(f"Total files to process: {len(image_files) + len(audio_files)}")
    
    # Create output directory
    os.makedirs(args.output_root, exist_ok=True)
    
    # Split files across GPUs
    image_splits = split_for_gpus(image_files, args.num_gpus)
    audio_splits = split_for_gpus(audio_files, args.num_gpus)
    
    # Start multiprocessing
    mp.set_start_method('spawn', force=True)
    
    # Process images
    print("\n" + "=" * 80)
    print("Encoding Images with DINOv2")
    print("=" * 80)
    
    processes = []
    manager = mp.Manager()
    progress_queue = manager.Queue()
    
    for gpu_id in range(args.num_gpus):
        img_paths = [x[0] for x in image_splits[gpu_id]]
        out_paths = [x[1] for x in image_splits[gpu_id]]
        
        p = mp.Process(target=encode_images_on_gpu, 
                      args=(gpu_id, img_paths, out_paths, args.dinov2_path, progress_queue))
        p.start()
        processes.append(p)
    
    # Monitor progress
    monitor_p = mp.Process(target=progress_monitor, args=(progress_queue, len(image_files)))
    monitor_p.start()
    
    for p in processes:
        p.join()
    
    monitor_p.terminate()
    monitor_p.join()
    
    print("Image encoding completed!")
    
    # Process audio
    print("\n" + "=" * 80)
    print("Encoding Audio with WavLM")
    print("=" * 80)
    
    processes = []
    progress_queue = manager.Queue()
    
    for gpu_id in range(args.num_gpus):
        audio_paths = [x[0] for x in audio_splits[gpu_id]]
        out_paths = [x[1] for x in audio_splits[gpu_id]]
        
        p = mp.Process(target=encode_audio_on_gpu,
                      args=(gpu_id, audio_paths, out_paths, args.wavlm_path, progress_queue))
        p.start()
        processes.append(p)
    
    # Monitor progress
    monitor_p = mp.Process(target=progress_monitor, args=(progress_queue, len(audio_files)))
    monitor_p.start()
    
    for p in processes:
        p.join()
    
    monitor_p.terminate()
    monitor_p.join()
    
    print("Audio encoding completed!")
    
    print("\n" + "=" * 80)
    print("Encoding Complete!")
    print("=" * 80)
    print(f"Encoded features saved to: {args.output_root}")
    print(f"Total images encoded: {len(image_files)}")
    print(f"Total audio files encoded: {len(audio_files)}")
    

if __name__ == '__main__':
    main()

