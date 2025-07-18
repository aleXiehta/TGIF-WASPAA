# data/datasets.py
import os
import csv
from collections import defaultdict
import random

import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader

from utils.helper import label_speaker_id


def get_dataloaders(config, is_ddp=False, world_size=1, rank=0):

    train_dataset = TGIFDataset(
        root_dir=config['dataset']['root_dir'],
        set_name='tr',
        segment_length=config['dataset']['segment_length'],
        sample_rate=config['dataset']['sample_rate'],
        transform=config['dataset']['transform']
    )

    val_dataset = TGIFDataset(
        root_dir=config['dataset']['root_dir'],
        set_name='cv',
        segment_length=config['dataset']['segment_length'],
        sample_rate=config['dataset']['sample_rate'],
        transform=config['dataset']['transform']
    )

    if is_ddp:
        from torch.utils.data.distributed import DistributedSampler
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)
        shuffle = False
    else:
        train_sampler = None
        val_sampler = None
        shuffle = True

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['train']['batch_size'],
        sampler=train_sampler,
        shuffle=True,
        num_workers=config['train']['num_workers'],
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['train']['batch_size'],
        sampler=val_sampler,
        shuffle=False,
        num_workers=config['train']['num_workers'],
        pin_memory=True
    )

    return train_loader, val_loader


def build_speaker_to_files_mapping(csv_file_path):
    """
    Build a mapping from speaker IDs to lists of file paths where they appear.
    """
    speaker_to_files = defaultdict(list)
    with open(csv_file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Extract the speaker IDs from the 'clean_file_path' field
            clean_file_path = row['clean_file_path']
            filename = os.path.basename(clean_file_path)
            if 'spkid_' in filename:
                # Get the substring after 'spkid_'
                spkid_str = filename.split('spkid_')[1]
                # Split the IDs by hyphen
                speaker_id = spkid_str.split('.')[0].split('-')[0]
                speaker_to_files[speaker_id].append(clean_file_path)
            else:
                # Handle cases where 'spkid_' is not in the filename
                continue
    return speaker_to_files

def get_enrollment_sample(speaker_to_files, current_file_path, subset='tr'):
    """
    Given the speaker-to-files mapping and the current file path,
    randomly select another file with the same speaker(s) as enrollment.
    """
    filename = os.path.basename(current_file_path)
    spkid_str = filename.split('spkid_')[1]
    speaker_id = spkid_str.split('.')[0].split('-')[0]

    # Get all files for this speaker
    files = speaker_to_files.get(speaker_id, [])
    # Exclude the current file
    if subset == 'tr':
        files = [f for f in files if f != current_file_path]
    if len(files) < 1:
        print(current_file_path)
    assert len(files) > 0
    enrollment_file = random.choice(files)
    return enrollment_file
    
class TGIFDataset(Dataset):
    def __init__(
            self, 
            root_dir, 
            set_name='tr', 
            segment_length=None, 
            enrollment_length=4, 
            sample_rate=16000, 
            transform=None, 
            **kwargs
        ):
        """
        Args:
            root_dir (string): Root directory of the TGIF-Dataset.
            set_name (string): Name of the dataset split ('tr' or 'cv').
            segment_length (int, optional): Randomly pick a few seconds of continuous chunk from the audio.
            sample_rate (int, optional): Desired sample rate. Must be <= 16000 Hz.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.set_name = set_name  # 'tr' for training, 'cv' for validation
        if segment_length is not None:
            self.segment_length = segment_length * sample_rate
        else:
            self.segment_length = segment_length

        self.enrollment_length = enrollment_length * sample_rate
        self.transform = transform
        self.sample_rate = sample_rate
        
        if self.sample_rate > 16000:
            raise RuntimeError("Sample rate must be less than or equal to 16 kHz.")
        
        # Construct the path to the metadata file
        metadata_file = os.path.join(root_dir, set_name, f'metadata.csv')
        if not os.path.isfile(metadata_file):
            raise FileNotFoundError(f"Metadata file not found at {metadata_file}")
        self.speaker_to_files = build_speaker_to_files_mapping(metadata_file)
        
        # Load the metadata
        self.metadata = pd.read_csv(metadata_file)
        if 'clean_file_path' not in self.metadata.columns or 'noisy_file_path' not in self.metadata.columns:
            raise ValueError("Metadata file must contain 'clean_file_path' and 'noisy_file_path' columns")
        self.metadata = label_speaker_id(self.metadata)
        
        # Get the file paths
        # self.clean_file_paths = self.metadata['clean_file_path'].tolist()
        # self.noisy_file_paths = self.metadata['noisy_file_path'].tolist()

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        # Get the relative file paths
        clean_path = self.metadata['clean_file_path'][idx]
        noisy_path = self.metadata['noisy_file_path'][idx]
        enrol_path = get_enrollment_sample(self.speaker_to_files, clean_path, subset=self.set_name)
        label = self.metadata['label'][idx]
        
        # Load the audio files
        clean_waveform, clean_sample_rate = torchaudio.load(clean_path)
        noisy_waveform, noisy_sample_rate = torchaudio.load(noisy_path)
        enrol_waveform, enrol_sample_rate = torchaudio.load(enrol_path)

        if self.segment_length is not None:
            original_length = clean_waveform.size(1)
            offset = torch.randint(0, original_length - self.segment_length + 1, size=(1,))
            enrol_offset = torch.randint(0, original_length - self.enrollment_length + 1, size=(1,))
            clean_waveform = clean_waveform[:, offset: offset + self.segment_length]
            noisy_waveform = noisy_waveform[:, offset: offset + self.segment_length]
            enrol_waveform = enrol_waveform[:, enrol_offset: enrol_offset + self.enrollment_length]
        
        # Ensure the sample rates are the same
        if clean_sample_rate != noisy_sample_rate:
            raise ValueError("Sample rates of clean and noisy audio files do not match")
        
        # Apply resampling if specified
        if self.sample_rate > 16000:
            raise RuntimeError("Sample rate must be less than or equal to 16 kHz.")
        if clean_sample_rate > self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=clean_sample_rate, new_freq=self.sample_rate)
            clean_waveform = resampler(clean_waveform)
            noisy_waveform = resampler(noisy_waveform)
            enrol_waveform = resampler(enrol_waveform)
            clean_sample_rate = self.sample_rate  # Update the sample rate
        
        # Apply transforms if any
        if self.transform:
            clean_waveform = self.transform(clean_waveform)
            noisy_waveform = self.transform(noisy_waveform)
            enrol_waveform = self.transform(enrol_waveform)
        
        sample = {
            'clean': clean_waveform,
            'noisy': noisy_waveform.squeeze(0),
            'enrollment': enrol_waveform.squeeze(0),
            'enrollment_length': enrol_waveform.size(-1),
            'clean_path': clean_path,
            'noisy_path': noisy_path,
            'enrollment_path': enrol_path,
            'label': label,
        }
        
        return sample
