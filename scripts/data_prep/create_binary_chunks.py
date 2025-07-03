#!/usr/bin/env python3
"""
Create pre-split binary chunks from ImageNet data for SystemDS LARS training.

This script reads existing CSV or binary data and splits it into manageable chunks
for memory-efficient training with large datasets.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

def create_binary_chunks(data_dir="imagenet_data", chunk_size=10000):
    """
    Create binary chunk files from existing ImageNet data.
    
    Args:
        data_dir: Directory containing the ImageNet data
        chunk_size: Number of samples per chunk
    """
    data_path = Path(data_dir)
    
    print(f"Creating binary chunks from data in: {data_path}")
    print(f"Chunk size: {chunk_size}")
    
    # Check what data we have available
    csv_train = data_path / "imagenet_train.csv"
    csv_val = data_path / "imagenet_val.csv"
    
    if csv_train.exists() and csv_val.exists():
        print("Found CSV files, converting to binary chunks...")
        create_chunks_from_csv(data_path, chunk_size)
    else:
        print("CSV files not found, creating dummy chunks for testing...")
        create_dummy_chunks(data_path, chunk_size)

def create_chunks_from_csv(data_path, chunk_size):
    """Create chunks from CSV files."""
    
    # Read training data
    print("Reading training CSV...")
    train_df = pd.read_csv(data_path / "imagenet_train.csv", header=None)
    print(f"Training data shape: {train_df.shape}")
    
    # Read validation data  
    print("Reading validation CSV...")
    val_df = pd.read_csv(data_path / "imagenet_val.csv", header=None)
    print(f"Validation data shape: {val_df.shape}")
    
    # Split training data into chunks
    train_labels = train_df.iloc[:, 0].values
    train_data = train_df.iloc[:, 1:].values
    
    # Convert to float and normalize
    train_data = train_data.astype(np.float64) / 255.0
    
    num_train_chunks = (len(train_data) + chunk_size - 1) // chunk_size
    print(f"Creating {num_train_chunks} training chunks...")
    
    for i in range(num_train_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(train_data))
        
        chunk_data = train_data[start_idx:end_idx]
        chunk_labels = train_labels[start_idx:end_idx]
        
        # Convert labels to one-hot (assuming 10 classes for now)
        num_classes = 10
        chunk_labels_onehot = np.eye(num_classes)[chunk_labels]
        
        # Save as binary files that SystemDS can read
        chunk_num = f"{i+1:03d}"
        
        # Save data chunk as CSV
        data_file = data_path / f"train_chunk_{chunk_num}.csv"
        pd.DataFrame(chunk_data).to_csv(data_file, header=False, index=False)
        
        # Save labels chunk as CSV
        labels_file = data_path / f"train_labels_{chunk_num}.csv"
        pd.DataFrame(chunk_labels_onehot).to_csv(labels_file, header=False, index=False)
        
        print(f"  Chunk {chunk_num}: {chunk_data.shape[0]} samples")
    
    # Process validation data (typically smaller, so fewer chunks)
    val_labels = val_df.iloc[:, 0].values
    val_data = val_df.iloc[:, 1:].values
    val_data = val_data.astype(np.float64) / 255.0
    
    val_chunk_size = min(chunk_size, len(val_data))
    num_val_chunks = (len(val_data) + val_chunk_size - 1) // val_chunk_size
    print(f"Creating {num_val_chunks} validation chunks...")
    
    for i in range(num_val_chunks):
        start_idx = i * val_chunk_size
        end_idx = min((i + 1) * val_chunk_size, len(val_data))
        
        chunk_data = val_data[start_idx:end_idx]
        chunk_labels = val_labels[start_idx:end_idx]
        
        # Convert labels to one-hot
        chunk_labels_onehot = np.eye(num_classes)[chunk_labels]
        
        chunk_num = f"{i+1:03d}"
        
        # Save data chunk as CSV
        data_file = data_path / f"val_chunk_{chunk_num}.csv"
        pd.DataFrame(chunk_data).to_csv(data_file, header=False, index=False)
        
        # Save labels chunk as CSV
        labels_file = data_path / f"val_labels_{chunk_num}.csv"
        pd.DataFrame(chunk_labels_onehot).to_csv(labels_file, header=False, index=False)
        
        print(f"  Val chunk {chunk_num}: {chunk_data.shape[0]} samples")

def create_dummy_chunks(data_path, chunk_size):
    """Create dummy chunks for testing when real data isn't available."""
    print("Creating dummy data chunks for testing...")
    
    # ImageNet-like dimensions
    img_height, img_width, channels = 224, 224, 3
    num_features = img_height * img_width * channels
    num_classes = 10
    
    # Create training chunks
    num_train_samples = chunk_size * 2  # Create 2 chunks for demo
    
    print(f"Generating {num_train_samples} dummy training samples...")
    train_data = np.random.rand(num_train_samples, num_features).astype(np.float64)
    train_labels = np.random.randint(0, num_classes, num_train_samples)
    train_labels_onehot = np.eye(num_classes)[train_labels]
    
    # Split into chunks
    for i in range(2):  # 2 training chunks
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size
        
        chunk_data = train_data[start_idx:end_idx]
        chunk_labels_onehot_chunk = train_labels_onehot[start_idx:end_idx]
        
        chunk_num = f"{i+1:03d}"
        
        # Save chunks as CSV
        data_file = data_path / f"train_chunk_{chunk_num}.csv"
        pd.DataFrame(chunk_data).to_csv(data_file, header=False, index=False)
        
        labels_file = data_path / f"train_labels_{chunk_num}.csv"
        pd.DataFrame(chunk_labels_onehot_chunk).to_csv(labels_file, header=False, index=False)
        
        print(f"  Created train chunk {chunk_num}: {chunk_data.shape}")
    
    # Create validation chunk
    num_val_samples = min(chunk_size, 5000)  # Smaller validation set
    print(f"Generating {num_val_samples} dummy validation samples...")
    
    val_data = np.random.rand(num_val_samples, num_features).astype(np.float64)
    val_labels = np.random.randint(0, num_classes, num_val_samples)
    val_labels_onehot = np.eye(num_classes)[val_labels]
    
    # Save validation chunk as CSV
    data_file = data_path / "val_chunk_001.csv"
    pd.DataFrame(val_data).to_csv(data_file, header=False, index=False)
    
    labels_file = data_path / "val_labels_001.csv"
    pd.DataFrame(val_labels_onehot).to_csv(labels_file, header=False, index=False)
    
    print(f"  Created val chunk 001: {val_data.shape}")

def main():
    """Main execution."""
    data_dir = "imagenet_data"
    chunk_size = 10000
    
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    if len(sys.argv) > 2:
        chunk_size = int(sys.argv[2])
    
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    create_binary_chunks(data_dir, chunk_size)
    
    print("\n✅ Binary chunk creation completed!")
    print(f"Chunks saved in: {data_dir}/")
    print("Files created:")
    
    data_path = Path(data_dir)
    for file in sorted(data_path.glob("*_chunk_*.bin")):
        size_mb = file.stat().st_size / (1024 * 1024)
        print(f"  {file.name} ({size_mb:.1f} MB)")

if __name__ == "__main__":
    main()