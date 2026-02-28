"""PFM file reading utilities for optical flow datasets.

PFM (Portable Float Map) is a format used by optical flow datasets like
Flying Chairs and ChairsSDHom for storing ground truth flow fields.
"""

import re
import numpy as np


def read_pfm(file_path: str) -> tuple[np.ndarray, float]:
    """Read a PFM file containing optical flow data.

    Args:
        file_path: Path to the .pfm file

    Returns:
        tuple of (flow_data, scale) where:
            - flow_data: numpy array of shape (H, W, 2) containing (x, y) flow components
            - scale: scale factor from the PFM header

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file is not a valid PFM file
    """
    with open(file_path, "rb") as file:
        # Read header
        header = file.readline().rstrip()
        if header.decode("ascii") == "PF":
            color = True
        elif header.decode("ascii") == "Pf":
            color = False
        else:
            raise ValueError(f"Not a PFM file: {file_path}")

        # Read dimensions
        dim_match = re.match(r"^(\d+)\s(\d+)\s$", file.readline().decode("ascii"))
        if not dim_match:
            raise ValueError(f"Malformed PFM header in: {file_path}")
        width, height = map(int, dim_match.groups())

        # Read scale and determine endianness
        scale = float(file.readline().decode("ascii").rstrip())
        if scale < 0:
            endian = "<"  # little endian
            scale = -scale
        else:
            endian = ">"  # big endian

        # Read data
        data = np.fromfile(file, endian + "f")

        # Reshape based on color/grayscale
        shape = (height, width, 3) if color else (height, width)
        data = np.reshape(data, shape)

        # Flip vertically (PFM stores rows bottom-to-top)
        data = np.flipud(data)

        # For flow, we only need first 2 channels (x, y)
        if color:
            data = data[..., :2]

        return data, scale


if __name__ == "__main__":
    # Test script
    import os

    # Find some PFM files in the dataset
    dataset_root = "datasets/ChairsSDHom/data"
    test_files = []

    for split in ["train", "test"]:
        flow_dir = os.path.join(dataset_root, split, "flow")
        if os.path.exists(flow_dir):
            files = [f for f in os.listdir(flow_dir) if f.endswith(".pfm")][:3]
            for f in files:
                test_files.append(os.path.join(flow_dir, f))

    print(f"Testing PFM reader on {len(test_files)} files...")

    for file_path in test_files:
        try:
            flow, scale = read_pfm(file_path)
            print(
                f"  {os.path.basename(file_path)}: shape={flow.shape}, "
                f"range=[{flow.min():.2f}, {flow.max():.2f}], scale={scale}"
            )
        except Exception as e:
            print(f"  ERROR reading {file_path}: {e}")

    print("\n--- TEST COMPLETE ---")
