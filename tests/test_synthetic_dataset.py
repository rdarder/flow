"""Tests for the synthetic dataset with multi-resolution support."""

try:
    import pytest
except ImportError:
    pytest = None  # pytest not required for direct execution

import torch
from torch.utils.data import DataLoader
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from synthetic_dataset import SyntheticFlowDataset


class TestSyntheticFlowDataset:
    """Test suite for SyntheticFlowDataset multi-resolution support."""
    
    def test_default_size_18x18(self):
        """Test that default size (18x18) still works."""
        dataset = SyntheticFlowDataset(img_size=18, length=10)
        img1, img2, flow = dataset[0]
        
        assert img1.shape == (18, 18, 3), f"Expected (18, 18, 3), got {img1.shape}"
        assert img2.shape == (18, 18, 3), f"Expected (18, 18, 3), got {img2.shape}"
        assert flow.shape == (18, 18, 2), f"Expected (18, 18, 2), got {flow.shape}"
    
    def test_32x32_resolution(self):
        """Test 32x32 resolution."""
        dataset = SyntheticFlowDataset(img_size=32, length=5)
        img1, img2, flow = dataset[0]
        
        assert img1.shape == (32, 32, 3)
        assert img2.shape == (32, 32, 3)
        assert flow.shape == (32, 32, 2)
    
    def test_64x64_resolution(self):
        """Test 64x64 resolution."""
        dataset = SyntheticFlowDataset(img_size=64, length=5)
        img1, img2, flow = dataset[0]
        
        assert img1.shape == (64, 64, 3)
        assert img2.shape == (64, 64, 3)
        assert flow.shape == (64, 64, 2)
    
    def test_batch_shapes(self):
        """Test that DataLoader produces correct batch shapes."""
        for img_size in [18, 32, 64]:
            dataset = SyntheticFlowDataset(img_size=img_size, length=10)
            loader = DataLoader(dataset, batch_size=4, shuffle=False)
            
            img1_batch, img2_batch, flow_batch = next(iter(loader))
            
            assert img1_batch.shape == (4, img_size, img_size, 3)
            assert img2_batch.shape == (4, img_size, img_size, 3)
            assert flow_batch.shape == (4, img_size, img_size, 2)
    
    def test_flow_field_dimensions_match_images(self):
        """Ensure flow field has same spatial dimensions as images."""
        for img_size in [16, 32, 64, 128]:
            dataset = SyntheticFlowDataset(img_size=img_size, length=3)
            for i in range(3):
                img1, img2, flow = dataset[i]
                assert flow.shape[:2] == img1.shape[:2] == (img_size, img_size)
    
    def test_motion_within_image_bounds(self):
        """Test that motion doesn't push blobs entirely out of image."""
        dataset = SyntheticFlowDataset(img_size=64, length=20)
        
        for i in range(20):
            img1, img2, flow = dataset[i]
            # Check that there's actually content in img2 (not all background)
            # This indirectly verifies motion kept blobs in bounds
            assert img2.mean() > 0.1, "img2 appears to be mostly empty"
    
    def test_no_nans(self):
        """Ensure no NaN values in generated data."""
        dataset = SyntheticFlowDataset(img_size=64, length=10)
        
        for i in range(10):
            img1, img2, flow = dataset[i]
            assert not torch.isnan(img1).any()
            assert not torch.isnan(img2).any()
            assert not torch.isnan(flow).any()
    
    def test_value_ranges(self):
        """Test that generated values are in expected ranges."""
        dataset = SyntheticFlowDataset(img_size=32, length=10)
        
        for i in range(10):
            img1, img2, flow = dataset[i]
            # Images should be in [0, 1]
            assert img1.min() >= 0 and img1.max() <= 1
            assert img2.min() >= 0 and img2.max() <= 1
            # Flow can be any float


if __name__ == "__main__":
    # Run tests
    test = TestSyntheticFlowDataset()
    test.test_default_size_18x18()
    print("✓ test_default_size_18x18 passed")
    
    test.test_32x32_resolution()
    print("✓ test_32x32_resolution passed")
    
    test.test_64x64_resolution()
    print("✓ test_64x64_resolution passed")
    
    test.test_batch_shapes()
    print("✓ test_batch_shapes passed")
    
    test.test_flow_field_dimensions_match_images()
    print("✓ test_flow_field_dimensions_match_images passed")
    
    test.test_motion_within_image_bounds()
    print("✓ test_motion_within_image_bounds passed")
    
    test.test_no_nans()
    print("✓ test_no_nans passed")
    
    test.test_value_ranges()
    print("✓ test_value_ranges passed")
    
    print("\n=== All tests passed! ===")
