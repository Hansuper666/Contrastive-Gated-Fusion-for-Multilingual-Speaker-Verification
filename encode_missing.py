import os
import torch
import numpy as np
import torchaudio
from transformers import Wav2Vec2FeatureExtractor, AutoModel
import warnings
import gc
warnings.filterwarnings('ignore')


def encode_audio_gpu(audio_path, output_path, model_path, gpu_id=0):
    """Encode audio on GPU with memory management"""
    print(f"\nProcessing: {os.path.basename(audio_path)}")
    
    device = torch.device(f'cuda:{gpu_id}')
    
    # Clear GPU cache before starting
    torch.cuda.empty_cache()
    gc.collect()
    
    # Load model
    print(f"  Loading WavLM model on GPU {gpu_id}...")
    processor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path).to(device)
    model.eval()
    
    with torch.no_grad():
        # Load audio
        waveform, sample_rate = torchaudio.load(audio_path)
        duration = waveform.shape[1] / sample_rate
        print(f"  Audio duration: {duration:.2f}s, size: {os.path.getsize(audio_path)/(1024**2):.2f} MB")
        
        # Resample if needed
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        waveform_np = waveform.squeeze().numpy()
        
        # Process entire audio (GPU should handle it if we clear memory first)
        print("  Processing on GPU...")
        inputs = processor(waveform_np, sampling_rate=16000, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        outputs = model(**inputs)
        features = outputs.last_hidden_state.cpu().numpy()
        print(f"  Feature shape: {features.shape}")
        
        # Clear GPU memory immediately
        del inputs, outputs, model
        torch.cuda.empty_cache()
        gc.collect()
        
        # Save features
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.save(output_path, features)
        
        # Check file size
        file_size = os.path.getsize(output_path)
        print(f"  Saved: {output_path}")
        print(f"  Output size: {file_size/(1024**2):.2f} MB")
        
        return True


def verify_all_files():
    """Verify all source files have been encoded"""
    print("\n" + "=" * 80)
    print("Verifying All Files Are Encoded")
    print("=" * 80)
    
    data_root = '/data/user_data/zeyangz/MAV-Celeb_v3'
    output_root = '/data/user_data/zeyangz/MAV-Celeb_v3/Pre-encoded'
    
    # Check images
    print("\nChecking images...")
    source_images = []
    
    # Training images
    faces_dir = os.path.join(data_root, 'faces')
    if os.path.exists(faces_dir):
        for root, dirs, files in os.walk(faces_dir):
            for file in files:
                if file.endswith('.jpg'):
                    source_images.append(os.path.join(root, file))
    
    # Test images
    for test_dir in ['English_test', 'German_test']:
        face_dir = os.path.join(data_root, test_dir, 'face')
        if os.path.exists(face_dir):
            for file in os.listdir(face_dir):
                if file.endswith('.jpg'):
                    source_images.append(os.path.join(face_dir, file))
    
    missing_images = []
    for img_path in source_images:
        rel_path = os.path.relpath(img_path, data_root)
        npy_path = os.path.join(output_root, rel_path.replace('.jpg', '.npy'))
        if not os.path.exists(npy_path):
            missing_images.append(img_path)
    
    print(f"  Total source images: {len(source_images)}")
    print(f"  Encoded images: {len(source_images) - len(missing_images)}")
    print(f"  Missing images: {len(missing_images)}")
    
    # Check audio
    print("\nChecking audio files...")
    source_audio = []
    
    # Training audio
    voices_dir = os.path.join(data_root, 'voices')
    if os.path.exists(voices_dir):
        for root, dirs, files in os.walk(voices_dir):
            for file in files:
                if file.endswith('.wav'):
                    source_audio.append(os.path.join(root, file))
    
    # Test audio
    for test_dir in ['English_test', 'German_test']:
        voice_dir = os.path.join(data_root, test_dir, 'voice')
        if os.path.exists(voice_dir):
            for file in os.listdir(voice_dir):
                if file.endswith('.wav'):
                    source_audio.append(os.path.join(voice_dir, file))
    
    missing_audio = []
    for audio_path in source_audio:
        rel_path = os.path.relpath(audio_path, data_root)
        npy_path = os.path.join(output_root, rel_path.replace('.wav', '.npy'))
        if not os.path.exists(npy_path):
            missing_audio.append(audio_path)
    
    print(f"  Total source audio: {len(source_audio)}")
    print(f"  Encoded audio: {len(source_audio) - len(missing_audio)}")
    print(f"  Missing audio: {len(missing_audio)}")
    
    # Summary
    print("\n" + "=" * 80)
    print("Verification Summary")
    print("=" * 80)
    print(f"Images: {len(source_images) - len(missing_images)}/{len(source_images)} encoded")
    print(f"Audio:  {len(source_audio) - len(missing_audio)}/{len(source_audio)} encoded")
    
    all_complete = len(missing_images) == 0 and len(missing_audio) == 0
    
    if all_complete:
        print("\nSUCCESS: All files have been encoded!")
    else:
        print("\nWARNING: Some files are missing:")
        if missing_images:
            print(f"\nMissing images ({len(missing_images)}):")
            for img in missing_images[:10]:
                print(f"  - {img}")
            if len(missing_images) > 10:
                print(f"  ... and {len(missing_images) - 10} more")
        
        if missing_audio:
            print(f"\nMissing audio ({len(missing_audio)}):")
            for aud in missing_audio[:10]:
                print(f"  - {aud}")
            if len(missing_audio) > 10:
                print(f"  ... and {len(missing_audio) - 10} more")
    
    return all_complete, missing_images, missing_audio


def main():
    data_root = '/data/user_data/zeyangz/MAV-Celeb_v3'
    output_root = '/data/user_data/zeyangz/MAV-Celeb_v3/Pre-encoded'
    wavlm_path = '/data/user_data/zeyangz/HiCMAE_LSMA/saved/model/pretrained/wavlm-large'
    
    # List of missing files
    missing_files = [
        '/data/user_data/zeyangz/MAV-Celeb_v3/voices/id0016/German/K8S0k0wdyx0/00000.wav',
        '/data/user_data/zeyangz/MAV-Celeb_v3/voices/id0018/German/ALX1fMj2kN4/00000.wav',
        '/data/user_data/zeyangz/MAV-Celeb_v3/voices/id0001/German/qnEJMy05qhU/00000.wav',
        '/data/user_data/zeyangz/MAV-Celeb_v3/voices/id0027/English/KrEkpowh490/00006.wav',
    ]
    
    print("=" * 80)
    print("Re-encoding Missing Audio Files")
    print("=" * 80)
    print(f"Total missing files: {len(missing_files)}")
    print("Using GPU with aggressive memory management")
    print("=" * 80)
    
    success_count = 0
    failed_count = 0
    
    for idx, audio_path in enumerate(missing_files, 1):
        print(f"\n[{idx}/{len(missing_files)}] Processing: {audio_path}")
        
        # Get output path
        rel_path = os.path.relpath(audio_path, data_root)
        output_path = os.path.join(output_root, rel_path.replace('.wav', '.npy'))
        
        # Alternate between GPUs
        gpu_id = idx % 2
        
        try:
            if encode_audio_gpu(audio_path, output_path, wavlm_path, gpu_id):
                success_count += 1
                print("  SUCCESS")
        except Exception as e:
            failed_count += 1
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("Re-encoding Complete!")
    print("=" * 80)
    print(f"Successfully encoded: {success_count}/{len(missing_files)}")
    print(f"Failed: {failed_count}/{len(missing_files)}")
    
    # Now verify all files
    verify_all_files()


if __name__ == '__main__':
    main()
