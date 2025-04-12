"""
Download a sample of N=10000 from a streaming dataset.
"""

import argparse
import os
from src.utils.data import download_sample
from datasets import load_dataset, Dataset, DatasetDict

def download_full(
        dataset: str
    ) -> Dataset:
    """
    Download the full dataset from the Hugging Face Hub.
    """

    print(f"Downloading full dataset from {dataset}...", flush=True)
    ds = load_dataset(dataset)
    splits = list(ds.keys())
    print(f"Dataset {dataset} has {len(splits)} splits: {splits}")
    assert len(splits) == 1, "More than one split present in the dataset. Exiting."

    # Generate train and test splits and save to disk
    single_split_name = splits[0]
    ds = ds[single_split_name]
    
    return ds
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--N", type=int, default=10000)
    parser.add_argument("--no_streaming", action="store_true", default=False)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--split_size", type=float, default=0.2)
    args = parser.parse_args()
    
    dataset_name = args.dataset
    test_size = args.split_size
    output_dir = args.output_dir
    N = args.N
    save_path = os.path.join(os.environ["HF_DATASETS_CACHE"], output_dir)
    
    if args.no_streaming:
        dataset = download_full(dataset_name)
    else:
        dataset = download_sample(dataset_name, N=N)

    print(dataset, flush=True)
    dataset_dict = dataset.train_test_split(test_size=test_size)
    print(f"Final dataset: {type(dataset_dict['train'])}", flush=True)
    print(f"Saving dataset to {save_path}...", flush=True)
    dataset_dict.save_to_disk(save_path)

if __name__ == "__main__":
    main()