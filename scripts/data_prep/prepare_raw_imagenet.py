#!/usr/bin/env python3
#-------------------------------------------------------------
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#
#-------------------------------------------------------------
"""
Raw ImageNet Data Preprocessing Pipeline
=========================================

This script processes raw ImageNet JPG images with metadata CSV files and prepares them
for SystemDS AlexNet training. It handles:

1. Reading metadata CSV files with file_path,label format
2. Loading JPG images (typically 256x256) and resizing to specified target size (default: 224x224)
3. Converting to normalized feature vectors
4. Creating one-hot encoded labels
5. Saving in SystemDS-compatible CSV format with resolution-based naming

Usage:
    python prepare_raw_imagenet.py --input_dir "C:/Users/romer/Desktop/Big_Data/imagenet/256x256" --output_dir "imagenet_data"
    python prepare_raw_imagenet.py --input_dir "C:/Users/romer/Desktop/Big_Data/imagenet/256x256" --target_size 299
    python prepare_raw_imagenet.py --input_dir "path/to/imagenet" --dry_run

Output files will be saved as:
    imagenet_data/<target_size>x<target_size>/imagenet_<target_size>x<target_size>_train.csv
    imagenet_data/<target_size>x<target_size>/imagenet_<target_size>x<target_size>_train_labels.csv
    imagenet_data/<target_size>x<target_size>/imagenet_<target_size>x<target_size>_test.csv
    imagenet_data/<target_size>x<target_size>/imagenet_<target_size>x<target_size>_test_labels.csv
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time
import gc
from PIL import Image
import csv

class RawImageNetProcessor:
    """Raw ImageNet JPG image processor for SystemDS."""
    
    def __init__(self, input_dir: str, output_dir: str = "imagenet_data/224x224", target_size: int = 224):
        self.input_dir = Path(input_dir)
        self.target_size = target_size
        
        # Create output directory based on resolution
        base_output = Path(output_dir).parent if "x" in Path(output_dir).name else Path(output_dir)
        self.output_dir = base_output / f"{target_size}x{target_size}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Target specifications for SystemDS AlexNet
        self.channels = 3
        self.features = self.target_size * self.target_size * self.channels
        self.num_classes = 1000
        
        print(f"Raw ImageNet Processor initialized")
        print(f"Input directory: {self.input_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Target format: {self.target_size}x{self.target_size}x{self.channels} images ({self.features} features), {self.num_classes} classes")
        print(f"Note: Source images will be resized from their original size to {self.target_size}x{self.target_size}")
    
    def inspect_raw_data(self) -> Dict:
        """Inspect the raw data structure and return metadata."""
        print("\n=== Raw Data Inspection ===")
        
        # Look for metadata files
        train_metadata_file = self.input_dir / "imagenet_train_metadata.csv"
        test_metadata_file = self.input_dir / "imagenet_test_metadata.csv"
        
        if not train_metadata_file.exists():
            raise FileNotFoundError(f"Training metadata file not found: {train_metadata_file}")
        if not test_metadata_file.exists():
            raise FileNotFoundError(f"Test metadata file not found: {test_metadata_file}")
        
        # Read metadata
        print(f"Reading training metadata from: {train_metadata_file}")
        train_df = pd.read_csv(train_metadata_file)
        print(f"Reading test metadata from: {test_metadata_file}")
        test_df = pd.read_csv(test_metadata_file)
        
        # Inspect structure
        print(f"\nTraining metadata shape: {train_df.shape}")
        print(f"Training columns: {list(train_df.columns)}")
        print(f"Training label range: {train_df['label'].min()} to {train_df['label'].max()}")
        print(f"Training unique labels: {train_df['label'].nunique()}")
        
        print(f"\nTest metadata shape: {test_df.shape}")
        print(f"Test columns: {list(test_df.columns)}")
        print(f"Test label range: {test_df['label'].min()} to {test_df['label'].max()}")
        print(f"Test unique labels: {test_df['label'].nunique()}")
        
        # Check if images actually exist
        print(f"\nChecking image availability...")
        train_available = self._count_available_images(train_df)
        test_available = self._count_available_images(test_df)
        
        # Sample a few images to check dimensions
        sample_dims = self._check_sample_image_dimensions(train_df.head(5))
        
        metadata = {
            'train_total': len(train_df),
            'train_available': train_available,
            'test_total': len(test_df),
            'test_available': test_available,
            'train_labels': sorted(train_df['label'].unique()),
            'test_labels': sorted(test_df['label'].unique()),
            'sample_dimensions': sample_dims
        }
        
        print(f"\n=== Summary ===")
        print(f"Training: {train_available}/{len(train_df)} images available")
        print(f"Test: {test_available}/{len(test_df)} images available")
        print(f"Sample image dimensions: {sample_dims}")
        
        return metadata
    
    def _count_available_images(self, df: pd.DataFrame) -> int:
        """Count how many images actually exist on disk."""
        available = 0
        total = len(df)
        
        print(f"  Checking {total} image files...")
        for i, row in df.iterrows():
            image_path = self.input_dir / row['file_path']
            if image_path.exists():
                available += 1
            
            # Progress update every 1000 images
            if (i + 1) % 1000 == 0:
                print(f"    Checked {i + 1}/{total} images, {available} available")
        
        print(f"  Final: {available}/{total} images available")
        return available
    
    def _check_sample_image_dimensions(self, sample_df: pd.DataFrame) -> List[Tuple]:
        """Check dimensions of a few sample images."""
        dimensions = []
        
        for _, row in sample_df.iterrows():
            image_path = self.input_dir / row['file_path']
            if image_path.exists():
                try:
                    with Image.open(image_path) as img:
                        dimensions.append((img.width, img.height, len(img.getbands())))
                except Exception as e:
                    print(f"    Error reading {image_path}: {e}")
            
            if len(dimensions) >= 3:  # Just check a few
                break
        
        return dimensions
    
    def process_dataset(self, max_samples: Optional[int] = None, dry_run: bool = False, skip_check: bool = False, split_from_train: bool = False) -> Dict:
        """Process the complete dataset."""
        print(f"\n=== Processing Dataset (dry_run={dry_run}) ===")
        
        # Read metadata
        train_df = pd.read_csv(self.input_dir / "imagenet_train_metadata.csv")
        
        if split_from_train:
            print("Creating validation set from training data...")
            # Skip test metadata entirely
            test_df = None
        else:
            test_df = pd.read_csv(self.input_dir / "imagenet_test_metadata.csv")
        
        # Filter to only available images (unless skipping)
        if not skip_check:
            print("Filtering to available images...")
            train_df = self._filter_available_images(train_df)
            if test_df is not None:
                test_df = self._filter_available_images(test_df)
        else:
            print("Skipping image availability check...")
        
        # Handle data splitting
        if split_from_train:
            # Use training data for both train and validation
            if max_samples:
                # Take first max_samples for training
                train_samples = max_samples
                # Use 20% of training samples for validation (or 400, whichever is smaller)
                val_samples = min(400, int(train_samples * 0.2), len(train_df) - train_samples)
                
                print(f"Splitting from training data:")
                print(f"  - Training: first {train_samples} samples")
                print(f"  - Validation: next {val_samples} samples")
                
                val_df = train_df.iloc[train_samples:train_samples + val_samples].copy()
                train_df = train_df.head(train_samples)
            else:
                # Default split: 90% train, 10% validation
                split_idx = int(len(train_df) * 0.9)
                val_df = train_df.iloc[split_idx:].copy()
                train_df = train_df.iloc[:split_idx].copy()
                print(f"Splitting training data: {len(train_df)} train, {len(val_df)} validation")
            
            test_df = val_df  # Use validation split as "test" for consistency
        else:
            # Limit samples if requested
            if max_samples:
                print(f"Limiting to {max_samples} samples per split...")
                train_df = train_df.head(max_samples)
                if test_df is not None:
                    test_df = test_df.head(max_samples)
        
        print(f"Processing {len(train_df)} training samples...")
        print(f"Processing {len(test_df)} test samples...")
        
        if dry_run:
            print("DRY RUN: Would process the above samples")
            return {'dry_run': True, 'train_samples': len(train_df), 'test_samples': len(test_df)}
        
        # Process training data
        train_results = self._process_split(train_df, "train")
        
        # Process test data (as validation)
        test_results = self._process_split(test_df, "val")
        
        return {
            'train': train_results,
            'validation': test_results
        }
    
    def _filter_available_images(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter dataframe to only include images that exist on disk."""
        available_mask = []
        
        for _, row in df.iterrows():
            image_path = self.input_dir / row['file_path']
            available_mask.append(image_path.exists())
        
        filtered_df = df[available_mask].copy()
        print(f"  Filtered {len(df)} -> {len(filtered_df)} available images")
        return filtered_df
    
    def _process_split(self, df: pd.DataFrame, split_name: str) -> Dict:
        """Process a data split (train or val)."""
        print(f"\nProcessing {split_name} split...")
        
        # Prepare output files with resolution in name
        # For val split, use 'test' in filename for consistency
        file_split_name = 'test' if split_name == 'val' else split_name
        features_file = self.output_dir / f"imagenet_{self.target_size}x{self.target_size}_{file_split_name}.csv"
        labels_file = self.output_dir / f"imagenet_{self.target_size}x{self.target_size}_{file_split_name}_labels.csv"
        
        # Process images in batches to manage memory
        batch_size = 1000
        total_samples = len(df)
        num_batches = (total_samples + batch_size - 1) // batch_size
        
        print(f"Processing {total_samples} samples in {num_batches} batches of {batch_size}")
        
        # Initialize CSV files
        features_written = 0
        labels_written = 0
        
        with open(features_file, 'w', newline='') as f_feat, \
             open(labels_file, 'w', newline='') as f_label:
            
            feat_writer = csv.writer(f_feat)
            label_writer = csv.writer(f_label)
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, total_samples)
                batch_df = df.iloc[start_idx:end_idx]
                
                print(f"  Batch {batch_idx + 1}/{num_batches}: Processing samples {start_idx}-{end_idx-1}")
                
                # Process batch
                batch_features, batch_labels = self._process_image_batch(batch_df)
                
                # Write to CSV
                for features_row in batch_features:
                    feat_writer.writerow(features_row)
                    features_written += 1
                
                for labels_row in batch_labels:
                    label_writer.writerow(labels_row)
                    labels_written += 1
                
                # Memory cleanup
                del batch_features, batch_labels
                gc.collect()
                
                print(f"    Wrote {len(batch_df)} samples to CSV")
        
        result = {
            'samples_processed': features_written,
            'features_file': str(features_file),
            'labels_file': str(labels_file),
            'features_shape': (features_written, self.features),
            'labels_shape': (labels_written, self.num_classes)
        }
        
        print(f"  {split_name} processing complete: {features_written} samples")
        return result
    
    def _process_image_batch(self, batch_df: pd.DataFrame) -> Tuple[List, List]:
        """Process a batch of images."""
        batch_features = []
        batch_labels = []
        
        for _, row in batch_df.iterrows():
            try:
                # Load and process image
                image_path = self.input_dir / row['file_path']
                features = self._process_single_image(image_path)
                
                # Process label
                label = int(row['label'])
                # Convert to 0-indexed if needed (ImageNet labels are usually 1-indexed)
                if label > 0:
                    label = label - 1
                
                # Create one-hot encoding
                one_hot = [0.0] * self.num_classes
                if 0 <= label < self.num_classes:
                    one_hot[label] = 1.0
                
                batch_features.append(features)
                batch_labels.append(one_hot)
                
            except Exception as e:
                print(f"    Error processing {row['file_path']}: {e}")
                # Skip this sample
                continue
        
        return batch_features, batch_labels
    
    def _process_single_image(self, image_path: Path) -> List[float]:
        """Process a single image: load, resize, normalize, flatten."""
        # Fix path if it points to wrong directory
        image_path_str = str(image_path)
        if "224x224" in image_path_str and "256x256" in str(self.input_dir):
            # Replace 224x224 with 256x256 in the path
            image_path_str = image_path_str.replace("224x224", "256x256")
            image_path = Path(image_path_str)
        
        # Load image
        with Image.open(image_path) as img:
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize to target size (e.g., from 256x256 to 224x224)
            if img.size != (self.target_size, self.target_size):
                img = img.resize((self.target_size, self.target_size), Image.LANCZOS)
            
            # Convert to numpy array and normalize to [0,1]
            img_array = np.array(img, dtype=np.float32) / 255.0
            
            # Flatten to feature vector
            features = img_array.flatten().tolist()
            
            return features


def main():
    parser = argparse.ArgumentParser(description='Process raw ImageNet JPG data for SystemDS')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing raw ImageNet data')
    parser.add_argument('--output_dir', type=str, default='imagenet_data',
                        help='Base output directory for processed data (resolution subdirs will be created)')
    parser.add_argument('--target_size', type=int, default=224,
                        help='Target image size (default: 224 for 224x224)')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples per split (for testing)')
    parser.add_argument('--dry_run', action='store_true',
                        help='Just inspect data without processing')
    parser.add_argument('--skip_check', action='store_true',
                        help='Skip image availability checking')
    parser.add_argument('--split_from_train', action='store_true',
                        help='Create validation set from training data instead of using test set')
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = RawImageNetProcessor(args.input_dir, args.output_dir, args.target_size)
    
    # Inspect data first (unless skipping check)
    if not args.skip_check:
        try:
            metadata = processor.inspect_raw_data()
        except Exception as e:
            print(f"Error during inspection: {e}")
            return 1
    else:
        print("Skipping data inspection...")
    
    # Process if not dry run
    if not args.dry_run:
        try:
            results = processor.process_dataset(
                max_samples=args.max_samples, 
                dry_run=False,
                skip_check=args.skip_check,
                split_from_train=args.split_from_train
            )
            print(f"\n=== Processing Complete ===")
            print(f"Results: {results}")
        except Exception as e:
            print(f"Error during processing: {e}")
            return 1
    else:
        processor.process_dataset(dry_run=True)
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 