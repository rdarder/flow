import torch
from torch.utils.data import Dataset
import os


class PreprocessedChairsDataset(Dataset):
    def __init__(self, path: str, split="train"):
        """
        A trivial dataset loader that loads a pre-processed
        list of tensors directly into RAM.
        """
        self.split = split
        self.filename = f"{path}/chairs_v0_{split}.pt"

        if not os.path.exists(self.filename):
            raise FileNotFoundError(
                f"File {self.filename} not found. "
                "Did you run `preprocess_chairs.py` first?"
            )

        print(f"Loading preprocessed data from {self.filename} into RAM...")
        # This one line loads the *entire* dataset into memory.
        self.data = torch.load(self.filename)
        print("Data loaded successfully.")

    def __len__(self):
        # The length is just the length of our list
        return len(self.data)

    def __getitem__(self, idx):
        # We just return the pre-processed tuple
        return self.data[idx]


if __name__ == "__main__":
    # --- Test Script ---
    print("Testing preprocessed data loader...")

    # This assumes you've already run preprocess_chairs.py
    try:
        train_dataset = PreprocessedChairsDataset(
            "../datasets/chairs_32", split="train"
        )

        # Test __len__
        print(f"Total training samples: {len(train_dataset)}")

        # Test __getitem__
        img1, img2, flow = train_dataset[0]

        print(f"\nSample 0 shapes:")
        print(f"  Img1 shape: {img1.shape} (Expected: 32, 32, 3)")
        print(f"  Flow shape: {flow.shape} (Expected: 64, 2)")

        assert img1.shape == (32, 32, 3)
        assert flow.shape == (64, 2)

        print("\n--- TEST PASSED ---")
        print("This dataset is ready to be used in your JAX training script.")

    except FileNotFoundError as e:
        print(f"\n--- TEST FAILED ---")
        print(e)
