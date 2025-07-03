#!/usr/bin/env python3
"""
Simple runner for raw ImageNet preprocessing
"""

import sys
import subprocess
from pathlib import Path

def main():
    # Default paths
    input_dir = r"C:\Users\romer\Desktop\Big_Data\imagenet\256x256"  # Source images are 256x256
    output_dir = "imagenet_data"
    
    print("Raw ImageNet Preprocessing Runner")
    print("=" * 50)
    print(f"Input directory: {input_dir} (256x256 source images)")
    print(f"Output directory: {output_dir}")
    print(f"Default target size: 224x224 (for AlexNet)")
    print()
    
    # Ask user what they want to do
    print("Choose an option:")
    print("1. Inspect data only (dry run)")
    print("2. Process small sample (2000 train + 400 val from training set)")
    print("3. Process full dataset (256x256 -> 224x224)")
    print("4. Process full dataset (256x256 -> custom size)")
    print("5. Custom processing")
    print()
    
    choice = input("Enter choice (1-5): ").strip()
    
    if choice == "1":
        # Dry run
        cmd = [
            sys.executable, "scripts/data_prep/prepare_raw_imagenet.py",
            "--input_dir", input_dir,
            "--output_dir", output_dir,
            "--dry_run"
        ]
    elif choice == "2":
        # Small sample with train/val split from training data
        print("Processing 2000 training + 400 validation samples from training set...")
        cmd = [
            sys.executable, "scripts/data_prep/prepare_raw_imagenet.py",
            "--input_dir", input_dir,
            "--output_dir", output_dir,
            "--max_samples", "2000",
            "--skip_check",
            "--split_from_train"
        ]
    elif choice == "3":
        # Full dataset 256x256 -> 224x224
        print("Processing 256x256 images -> 224x224 for AlexNet...")
        cmd = [
            sys.executable, "scripts/data_prep/prepare_raw_imagenet.py",
            "--input_dir", input_dir,
            "--output_dir", output_dir,
            "--target_size", "224",
            "--skip_check"
        ]
    elif choice == "4":
        # Full dataset custom resolution
        target_size = input("Enter target size (e.g., 256, 299): ").strip()
        if not target_size.isdigit():
            print("Invalid target size!")
            return 1
        
        print(f"Processing 256x256 images -> {target_size}x{target_size}...")
        
        cmd = [
            sys.executable, "scripts/data_prep/prepare_raw_imagenet.py",
            "--input_dir", input_dir,
            "--output_dir", output_dir,
            "--target_size", target_size,
            "--skip_check"
        ]
    elif choice == "5":
        # Custom
        custom_input = input(f"Input directory [{input_dir}]: ").strip()
        if custom_input:
            input_dir = custom_input
        
        custom_output = input(f"Output directory [{output_dir}]: ").strip()
        if custom_output:
            output_dir = custom_output
        
        target_size = input("Target size [224]: ").strip() or "224"
        max_samples = input("Max samples per split (leave empty for all): ").strip()
        
        cmd = [
            sys.executable, "scripts/data_prep/prepare_raw_imagenet.py",
            "--input_dir", input_dir,
            "--output_dir", output_dir,
            "--target_size", target_size
        ]
        
        if max_samples:
            cmd.extend(["--max_samples", max_samples])
        
        skip_check = input("Skip image availability check? [Y/n]: ").strip().lower()
        if skip_check != 'n':
            cmd.append("--skip_check")
            
        split_from_train = input("Create validation from training data? [y/N]: ").strip().lower()
        if split_from_train == 'y':
            cmd.append("--split_from_train")
    else:
        print("Invalid choice!")
        return 1
    
    print(f"\nRunning command: {' '.join(cmd)}")
    print()
    
    # Run the command
    try:
        result = subprocess.run(cmd, check=True)
        print("\nProcessing completed successfully!")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"\nError during processing: {e}")
        return 1
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 