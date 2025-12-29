#!/usr/bin/env python3
"""
Analyze ShanghaiTech A density distribution to calculate optimal bin centers.

This script analyzes the ground truth density maps to understand the distribution
of person counts per block at different reduction factors (8, 16, 32).

Usage:
    python analyze_sha_bins.py --data_root ./data/sha --reduction 16
"""

import os
import argparse
import numpy as np
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter
from PIL import Image
from collections import Counter
import matplotlib.pyplot as plt


def load_points_from_mat(mat_path):
    """Load point annotations from .mat file."""
    mat = loadmat(mat_path)
    # Try different possible key names
    for key in ['image_info', 'annPoints', 'points']:
        if key in mat:
            if key == 'image_info':
                points = mat[key][0, 0]['location'][0, 0]
            else:
                points = mat[key]
            return points
    
    # Try to find any array that looks like points
    for key, value in mat.items():
        if not key.startswith('_') and isinstance(value, np.ndarray):
            if len(value.shape) == 2 and value.shape[1] == 2:
                return value
    
    raise ValueError(f"Could not find points in {mat_path}")


def count_per_block(points, img_shape, block_size):
    """
    Count number of people in each block.
    
    Args:
        points: Nx2 array of (x, y) coordinates
        img_shape: (H, W) of the image
        block_size: Size of each block (e.g., 8, 16)
    
    Returns:
        2D array of counts per block
    """
    H, W = img_shape
    h_blocks = H // block_size
    w_blocks = W // block_size
    
    counts = np.zeros((h_blocks, w_blocks), dtype=np.int32)
    
    if len(points) == 0:
        return counts
    
    for x, y in points:
        # Convert to block indices
        bx = int(x) // block_size
        by = int(y) // block_size
        
        # Check bounds
        if 0 <= bx < w_blocks and 0 <= by < h_blocks:
            counts[by, bx] += 1
    
    return counts


def analyze_dataset(data_root, reduction, split='train'):
    """
    Analyze the distribution of counts per block in the dataset.
    """
    if 'sha' in data_root.lower():
        img_dir = os.path.join(data_root, split, 'images')
        gt_dir = os.path.join(data_root, split, 'labels')
    else:
        # Standard ShanghaiTech structure
        img_dir = os.path.join(data_root, f'{split}_data', 'images')
        gt_dir = os.path.join(data_root, f'{split}_data', 'labels')
    
    if not os.path.exists(img_dir):
        # Try alternative paths
        for alt_img in ['images', 'img', 'imgs']:
            alt_path = os.path.join(data_root, split, alt_img)
            if os.path.exists(alt_path):
                img_dir = alt_path
                break
    
    if not os.path.exists(gt_dir):
        for alt_gt in ['ground_truth', 'ground-truth', 'gt', 'GT']:
            alt_path = os.path.join(data_root, split, alt_gt)
            if os.path.exists(alt_path):
                gt_dir = alt_path
                break
    
    print(f"Image directory: {img_dir}")
    print(f"GT directory: {gt_dir}")
    
    if not os.path.exists(img_dir) or not os.path.exists(gt_dir):
        raise FileNotFoundError(f"Could not find data directories")
    
    all_counts = []
    total_people = 0
    total_blocks = 0
    
    img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
    
    print(f"\nAnalyzing {len(img_files)} images with block size {reduction}...")
    
    for img_file in img_files:
        # Load image to get shape
        img_path = os.path.join(img_dir, img_file)
        img = Image.open(img_path)
        W, H = img.size
        
        # Find corresponding GT file
        base_name = os.path.splitext(img_file)[0]
        gt_file = None
        for gt_pattern in [f'GT_{base_name}.mat', f'{base_name}.mat', f'{base_name}_ann.mat']:
            gt_path = os.path.join(gt_dir, gt_pattern)
            if os.path.exists(gt_path):
                gt_file = gt_path
                break
        
        if gt_file is None:
            print(f"Warning: No GT found for {img_file}")
            continue
        
        # Load points
        try:
            points = load_points_from_mat(gt_file)
        except Exception as e:
            print(f"Error loading {gt_file}: {e}")
            continue
        
        # Count per block
        counts = count_per_block(points, (H, W), reduction)
        
        all_counts.extend(counts.flatten().tolist())
        total_people += len(points)
        total_blocks += counts.size
    
    all_counts = np.array(all_counts)
    
    return all_counts, total_people, total_blocks


def calculate_bin_centers(counts, num_bins=5, strategy='fine'):
    """
    Calculate optimal bin centers based on distribution.
    
    Args:
        counts: Array of all block counts
        num_bins: Number of bins to use
        strategy: 'fine', 'dynamic', or 'coarse'
    
    Returns:
        bin_edges: List of bin edges
        bin_centers: List of bin centers (anchor points)
    """
    max_count = int(counts.max())
    
    print(f"\n{'='*60}")
    print(f"Distribution Analysis")
    print(f"{'='*60}")
    print(f"Total blocks: {len(counts)}")
    print(f"Max count per block: {max_count}")
    print(f"Mean count per block: {counts.mean():.4f}")
    print(f"Median count per block: {np.median(counts):.4f}")
    print(f"Std count per block: {counts.std():.4f}")
    
    # Count distribution
    counter = Counter(counts)
    print(f"\nCount distribution (top 20):")
    for count, freq in sorted(counter.items())[:20]:
        pct = 100.0 * freq / len(counts)
        print(f"  Count {count:3d}: {freq:7d} blocks ({pct:5.2f}%)")
    
    # Calculate percentiles
    percentiles = [50, 75, 90, 95, 99, 99.9]
    print(f"\nPercentiles:")
    for p in percentiles:
        val = np.percentile(counts, p)
        print(f"  {p:5.1f}%: {val:.2f}")
    
    # Calculate bin edges and centers based on strategy
    if strategy == 'fine':
        # Fine: each integer gets its own bin, last bin is [m, ∞)
        # Find truncation point (m) where we want to merge
        # Usually set so that the last bin has reasonable sample size
        
        # Find m such that counts >= m represent ~5% of non-zero blocks
        non_zero_counts = counts[counts > 0]
        if len(non_zero_counts) > 0:
            m = int(np.percentile(non_zero_counts, 95))
            m = max(m, num_bins - 1)  # At least num_bins - 1
        else:
            m = num_bins - 1
        
        # Bins: {0}, {1}, {2}, ..., {m-1}, [m, ∞)
        bin_edges = list(range(num_bins - 1)) + [m]
        
        # Centers: for integer bins, center = value
        # For last bin, center = average of all counts >= m
        bin_centers = list(range(num_bins - 1))
        last_bin_counts = counts[counts >= (num_bins - 1)]
        if len(last_bin_counts) > 0:
            last_center = last_bin_counts.mean()
        else:
            last_center = num_bins - 1
        bin_centers.append(last_center)
        
    elif strategy == 'quantile':
        # Quantile-based: equal number of samples in each bin
        quantiles = np.linspace(0, 100, num_bins + 1)
        bin_edges = [np.percentile(counts, q) for q in quantiles]
        bin_centers = [(bin_edges[i] + bin_edges[i+1]) / 2 for i in range(num_bins)]
        
    else:  # dynamic or custom
        # Based on the actual distribution, create meaningful bins
        # For crowd counting, most blocks are empty
        
        # Strategy: 0, 1, 2-3, 4-7, 8+
        if num_bins == 5:
            bin_edges = [0, 1, 2, 4, 8]
            bin_centers = []
            
            # Calculate actual centers based on data
            # Bin 0: exactly 0
            bin_centers.append(0.0)
            
            # Bin 1: exactly 1
            bin_centers.append(1.0)
            
            # Bin 2: 2-3
            mask = (counts >= 2) & (counts <= 3)
            if mask.sum() > 0:
                bin_centers.append(counts[mask].mean())
            else:
                bin_centers.append(2.5)
            
            # Bin 3: 4-7
            mask = (counts >= 4) & (counts <= 7)
            if mask.sum() > 0:
                bin_centers.append(counts[mask].mean())
            else:
                bin_centers.append(5.5)
            
            # Bin 4: 8+
            mask = counts >= 8
            if mask.sum() > 0:
                bin_centers.append(counts[mask].mean())
            else:
                bin_centers.append(8.0)
    
    return bin_edges, bin_centers


def suggest_config(counts, num_bins=5):
    """
    Suggest configuration based on analysis.
    """
    print(f"\n{'='*60}")
    print(f"RECOMMENDED CONFIGURATION")
    print(f"{'='*60}")
    
    # Strategy 1: Fine-grained (like original paper)
    print("\n📌 Option 1: Fine-grained bins (like CLIP-EBC paper)")
    print("   Bins: {0}, {1}, {2}, {3}, [4+]")
    
    # Calculate last bin center
    last_counts = counts[counts >= 4]
    if len(last_counts) > 0:
        last_center = min(last_counts.mean(), 10)  # Cap at 10
    else:
        last_center = 4.0
    
    centers_fine = [0.0, 1.0, 2.0, 3.0, round(last_center, 1)]
    print(f"   BIN_CENTERS: {centers_fine}")
    
    # Strategy 2: Adaptive based on data
    print("\n📌 Option 2: Adaptive bins based on SHA distribution")
    
    # Most blocks are empty, so we want:
    # - One bin for 0
    # - One bin for 1
    # - One bin for 2
    # - One bin for 3-4
    # - One bin for 5+
    
    p95 = np.percentile(counts[counts > 0], 95) if (counts > 0).sum() > 0 else 5
    
    centers_adaptive = [0.0, 1.0, 2.0]
    
    # Bin for 3-4
    mask = (counts >= 3) & (counts <= 4)
    if mask.sum() > 0:
        centers_adaptive.append(round(counts[mask].mean(), 1))
    else:
        centers_adaptive.append(3.5)
    
    # Bin for 5+
    mask = counts >= 5
    if mask.sum() > 0:
        centers_adaptive.append(round(min(counts[mask].mean(), 8), 1))
    else:
        centers_adaptive.append(5.0)
    
    print(f"   BIN_CENTERS: {centers_adaptive}")
    
    # Strategy 3: Conservative
    print("\n📌 Option 3: Conservative (safer for training)")
    centers_conservative = [0.0, 1.0, 2.0, 3.0, 4.0]
    print(f"   BIN_CENTERS: {centers_conservative}")
    
    print("\n" + "="*60)
    print("YAML CONFIG UPDATE:")
    print("="*60)
    print(f"""
# For reduction=16 on ShanghaiTech A
MODEL:
  NUM_BINS: 5
  BIN_CENTERS: {centers_fine}  # Option 1 (recommended)
  # BIN_CENTERS: {centers_adaptive}  # Option 2
  # BIN_CENTERS: {centers_conservative}  # Option 3
""")
    
    # Also show text prompts
    print("\nCorresponding text prompts:")
    for i, center in enumerate(centers_fine[:-1]):
        c = int(center)
        if c == 0:
            print(f"  Bin {i}: 'There is no person.'")
        elif c == 1:
            print(f"  Bin {i}: 'There is one person.'")
        else:
            print(f"  Bin {i}: 'There are {c} people.'")
    
    last_center = int(centers_fine[-1])
    print(f"  Bin {len(centers_fine)-1}: 'There are more than {int(centers_fine[-2])} people.'")
    
    return centers_fine, centers_adaptive, centers_conservative


def main():
    parser = argparse.ArgumentParser(description='Analyze SHA distribution for bin centers')
    parser.add_argument('--data_root', type=str, default='./data/sha',
                        help='Path to ShanghaiTech A data')
    parser.add_argument('--reduction', type=int, default=16,
                        help='Block size / reduction factor')
    parser.add_argument('--num_bins', type=int, default=5,
                        help='Number of bins')
    parser.add_argument('--split', type=str, default='train',
                        help='Which split to analyze')
    args = parser.parse_args()
    
    print(f"Analyzing ShanghaiTech A with reduction={args.reduction}")
    print(f"Data root: {args.data_root}")
    
    try:
        counts, total_people, total_blocks = analyze_dataset(
            args.data_root, 
            args.reduction,
            args.split
        )
        
        print(f"\nTotal people: {total_people}")
        print(f"Total blocks: {total_blocks}")
        print(f"Average people per block: {total_people/total_blocks:.4f}")
        
        # Suggest configurations
        suggest_config(counts, args.num_bins)
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nPlease make sure the data directory structure is correct.")
        print("Expected structure:")
        print("  data/sha/train/images/")
        print("  data/sha/train/labels/")


if __name__ == '__main__':
    main()