import os
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


class MAVCelebEncodedDataset(Dataset):
    """
    Dataset for pre-encoded MAV-Celeb features
    """
    def __init__(
        self,
        data_file,
        encoded_root,
        data_root='/data/user_data/zeyangz/MAV-Celeb_v3',
        is_train=True
    ):
        """
        Args:
            data_file: Path to txt file with audio-image pairs
            encoded_root: Root directory with pre-encoded features
            data_root: Original data root
            is_train: True for training, False for testing
        """
        self.encoded_root = encoded_root
        self.data_root = data_root
        self.is_train = is_train
        
        # Read data file
        self.samples = []
        self.labels = []
        self.keys = []
        
        with open(data_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                
                if is_train:
                    # Training format: audio_path image_path
                    audio_rel_path = parts[0]
                    image_rel_path = parts[1]
                    
                    # Extract person ID from path (e.g., id0001)
                    # id0001 -> 1 -> subtract 1 -> 0 (for 0-based indexing)
                    # id0050 -> 50 -> subtract 1 -> 49
                    person_id = audio_rel_path.split('/')[0]
                    id_num = int(person_id.replace('id', '').lstrip('0') or '0')
                    label = id_num - 1  # Convert to 0-based for CrossEntropyLoss
                    key = None
                else:
                    # Test format: key audio_path image_path
                    key = parts[0]
                    audio_rel_path = parts[1]
                    image_rel_path = parts[2]
                    label = -1  # Unknown for test
                
                self.samples.append((audio_rel_path, image_rel_path))
                self.labels.append(label)
                self.keys.append(key)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        audio_rel_path, image_rel_path = self.samples[idx]
        label = self.labels[idx]
        key = self.keys[idx]
        
        # Map paths to encoded directory structure
        if self.is_train:
            # Training: id0001/English/video/00000.wav -> voices/id0001/English/video/00000.npy
            audio_npy_path = os.path.join(
                self.encoded_root,
                'voices',
                audio_rel_path.replace('.wav', '.npy')
            )
            # Training: id0001/English/video/000000100.jpg -> faces/id0001/English/video/000000100.npy
            image_npy_path = os.path.join(
                self.encoded_root,
                'faces',
                image_rel_path.replace('.jpg', '.npy')
            )
        else:
            # Test: English_test/voices/0000.wav -> English_test/voice/0000.npy
            audio_npy_path = os.path.join(
                self.encoded_root,
                audio_rel_path.replace('.wav', '.npy').replace('/voices/', '/voice/')
            )
            # Test: English_test/faces/0000.jpg -> English_test/face/0000.npy
            image_npy_path = os.path.join(
                self.encoded_root,
                image_rel_path.replace('.jpg', '.npy').replace('/faces/', '/face/')
            )
        
        # Load encoded features
        try:
            audio_features = np.load(audio_npy_path)  # Shape: (1, T, 1024)
            image_features = np.load(image_npy_path)  # Shape: (1, 257, 1536)
            
            # Average pooling to get single vector
            audio_vector = audio_features.mean(axis=1).squeeze()  # (1024,)
            image_vector = image_features.mean(axis=1).squeeze()  # (1536,)
            
            # Convert to tensors
            audio_tensor = torch.from_numpy(audio_vector).float()
            image_tensor = torch.from_numpy(image_vector).float()
            
            if self.is_train:
                return audio_tensor, image_tensor, label
            else:
                return audio_tensor, image_tensor, label, key
                
        except Exception as e:
            print(f"Error loading {audio_npy_path} or {image_npy_path}: {e}")
            raise


class MAVCelebEncodedTestDataset(Dataset):
    """
    Test dataset that returns keys for verification evaluation
    """
    def __init__(self, data_file, encoded_root, data_root='/data/user_data/zeyangz/MAV-Celeb_v3'):
        self.dataset = MAVCelebEncodedDataset(
            data_file=data_file,
            encoded_root=encoded_root,
            data_root=data_root,
            is_train=False
        )
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]

