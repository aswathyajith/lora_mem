"""
Download a sample of N=10000 from a streaming dataset.
"""

import argparse
import os
from src.utils.data import download_sample

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--N", type=int, default=10000)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--split_size", type=float, default=0.2)
    args = parser.parse_args()
    
    dataset_name = args.dataset
    test_size = args.split_size
    output_dir = args.output_dir
    N = args.N
    dataset = download_sample(dataset_name, N=N)
    dataset = dataset.train_test_split(test_size=test_size)
    dataset.save_to_disk(os.path.join(os.environ["HF_DATASETS_CACHE"], output_dir))

if __name__ == "__main__":
    main()