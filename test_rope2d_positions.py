#!/usr/bin/env python3
"""Test RoPE2D correctness with TREAD routing."""

import torch
from model import TokenRouter

def test_rope2d_positions():
    """Verify that token positions are computed correctly for routed tokens."""
    print("=" * 60)
    print("Testing RoPE2D Position Computation with TREAD")
    print("=" * 60)
    
    # Create router
    router = TokenRouter(
        start_layer=2,
        end_layer=8,
        drop_percent=50,
        total_layers=12
    )
    
    # Create input with 16x16 patch grid (256 tokens)
    batch_size = 2
    num_cls_tokens = 16  # registers
    height, width = 16, 16
    num_patches = height * width
    hidden_dim = 256
    
    x = torch.randn(batch_size, num_cls_tokens + num_patches, hidden_dim)
    patch_shape = (height, width)
    
    print("\n[Test 1] Verify route_tokens_from_start computes 2D positions...")
    route_state = router.route_tokens_from_start(x, patch_shape, num_cls_tokens)
    
    direct_positions_2d = route_state['direct_positions_2d']
    direct_indices = route_state['direct_indices']
    
    print(f"✓ Positions computed: shape {direct_positions_2d.shape}")
    print(f"  - Number of direct tokens: {len(direct_indices)}")
    print(f"  - Position tensor shape: {direct_positions_2d.shape}")
    
    # Verify positions are correct
    print("\n[Test 2] Verify computed positions match original indices...")
    for i, flat_idx in enumerate(direct_indices):
        h_true = flat_idx.item() // width
        w_true = flat_idx.item() % width
        h_computed = int(direct_positions_2d[i, 0].item())
        w_computed = int(direct_positions_2d[i, 1].item())
        assert h_true == h_computed, f"Height mismatch at {i}: {h_true} != {h_computed}"
        assert w_true == w_computed, f"Width mismatch at {i}: {w_true} != {w_computed}"
    
    print("✓ All positions computed correctly")
    
    # Test that positions are in valid range
    print("\n[Test 3] Verify positions are in valid grid range...")
    assert direct_positions_2d[:, 0].min() >= 0 and direct_positions_2d[:, 0].max() < height, \
        f"Height out of range: [{direct_positions_2d[:, 0].min()}, {direct_positions_2d[:, 0].max()}]"
    assert direct_positions_2d[:, 1].min() >= 0 and direct_positions_2d[:, 1].max() < width, \
        f"Width out of range: [{direct_positions_2d[:, 1].min()}, {direct_positions_2d[:, 1].max()}]"
    print("✓ All positions in valid grid range")
    
    # Test that positions are unique (no duplicates)
    print("\n[Test 4] Verify no duplicate positions...")
    unique_positions = torch.unique(direct_positions_2d, dim=0)
    assert len(unique_positions) == len(direct_positions_2d), \
        f"Found duplicate positions: {len(direct_positions_2d)} tokens, {len(unique_positions)} unique"
    print("✓ All positions are unique")
    
    # Test PoMMixer can use these positions
    print("\n[Test 5] Verify PoMMixer can use token positions...")
    try:
        from pom import PoMMixer
        
        mixer = PoMMixer(dim=hidden_dim, degree=3, expand=2)
        mixer.eval()
        
        # Prepare input: add cls tokens back
        x_cls = torch.randn(batch_size, num_cls_tokens, hidden_dim)
        x_patches = torch.randn(batch_size, len(direct_indices), hidden_dim)
        x_input = torch.cat([x_cls, x_patches], dim=1)
        
        # Test forward with token positions
        with torch.no_grad():
            # Note: move to GPU if available
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            mixer = mixer.to(device)
            x_input = x_input.to(device)
            positions = direct_positions_2d.to(device)
            
            # Forward pass should work with token_positions
            output = mixer(x_input, patch_shape=patch_shape, num_cls_tokens=num_cls_tokens,
                          token_positions=positions)
        
        print("✓ PoMMixer forward pass with token_positions successful")
        print(f"  - Output shape: {output.shape}")
        assert output.shape == x_input.shape, f"Shape mismatch: {output.shape} != {x_input.shape}"
        print("✓ Output shape matches input")
        
    except Exception as e:
        print(f"✗ PoMMixer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    print("✓ ALL RoPE2D POSITION TESTS PASSED!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    import sys
    success = test_rope2d_positions()
    sys.exit(0 if success else 1)
