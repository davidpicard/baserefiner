#!/usr/bin/env python3
"""Test script for TREAD implementation."""

import torch
import sys
from model import Baseline

def test_tread_basic():
    """Test basic TREAD functionality."""
    print("=" * 60)
    print("Testing TREAD Token Routing Implementation")
    print("=" * 60)
    
    # Test 1: Create model WITH TREAD
    print("\n[Test 1] Creating model WITH TREAD enabled...")
    try:
        model_with_tread = Baseline(
            input_size=32,
            patch_size=2,
            hidden_dim=256,
            depth=12,
            num_heads=8,
            emb_dim=256,
            use_tread=True,
            tread_start_layer=2,
            tread_end_layer=8,
            tread_drop_percent=50
        )
        print("✓ Model with TREAD created successfully")
        print(f"  - TREAD enabled: {model_with_tread.use_tread}")
        print(f"  - Route layers: {model_with_tread.tread_start_layer} → {model_with_tread.tread_end_layer}")
        print(f"  - Drop percent: {model_with_tread.tread_drop_percent}%")
        print(f"  - Selection rate: {model_with_tread.token_router.selection_rate*100:.1f}%")
    except Exception as e:
        print(f"✗ Failed to create model with TREAD: {e}")
        return False
    
    # Test 2: Create model WITHOUT TREAD  
    print("\n[Test 2] Creating model WITHOUT TREAD...")
    try:
        model_baseline = Baseline(
            input_size=32,
            patch_size=2,
            hidden_dim=256,
            depth=12,
            num_heads=8,
            emb_dim=256,
            use_tread=False
        )
        print("✓ Baseline model created successfully")
        print(f"  - TREAD enabled: {model_baseline.use_tread}")
        print(f"  - Token router: {model_baseline.token_router}")
    except Exception as e:
        print(f"✗ Failed to create baseline model: {e}")
        return False
    
    # Test 3: Forward pass with TREAD
    print("\n[Test 3] Forward pass with TREAD...")
    try:
        batch_size = 2
        x = torch.randn(batch_size, 3, 32, 32)
        t = torch.rand(batch_size)
        
        # Move to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_with_tread = model_with_tread.to(device)
        x = x.to(device)
        t = t.to(device)
        
        model_with_tread.eval()
        with torch.no_grad():
            output = model_with_tread(x, t)
        
        print("✓ Forward pass with TREAD successful")
        print(f"  - Input shape: {x.shape}")
        print(f"  - Output shape: {output.shape}")
        assert output.shape == x.shape, f"Output shape {output.shape} != input shape {x.shape}"
        print("✓ Output shape matches input shape")
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 4: Forward pass without TREAD
    print("\n[Test 4] Forward pass without TREAD...")
    try:
        x = torch.randn(batch_size, 3, 32, 32)
        t = torch.rand(batch_size)
        
        # Move to GPU if available
        model_baseline = model_baseline.to(device)
        x = x.to(device)
        t = t.to(device)
        
        model_baseline.eval()
        with torch.no_grad():
            output = model_baseline(x, t)
        
        print("✓ Forward pass without TREAD successful")
        print(f"  - Input shape: {x.shape}")
        print(f"  - Output shape: {output.shape}")
        assert output.shape == x.shape, f"Output shape {output.shape} != input shape {x.shape}"
        print("✓ Output shape matches input shape")
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 5: Verify TREAD configuration in model
    print("\n[Test 5] Testing TREAD parameters are configurable...")
    try:
        # Create a Baseline model with different TREAD settings
        config = {
            'input_size': 32,
            'patch_size': 2,
            'hidden_dim': 256,
            'depth': 12,
            'num_heads': 8,
            'emb_dim': 256,
            'use_tread': True,
            'tread_start_layer': 3,
            'tread_end_layer': 10,
            'tread_drop_percent': 25
        }
        
        model = Baseline(**config)
        print("✓ Model with custom TREAD settings created successfully")
        print(f"  - Start layer: {model.tread_start_layer}")
        print(f"  - End layer: {model.tread_end_layer}")
        print(f"  - Drop percent: {model.tread_drop_percent}%")
        print(f"  - Selection rate: {model.token_router.selection_rate*100:.1f}%")
        
        x = torch.randn(2, 3, 32, 32)
        t = torch.rand(2)
        
        # Move to device
        model = model.to(device)
        x = x.to(device)
        t = t.to(device)
        
        model.eval()
        with torch.no_grad():
            output = model(x, t)
        
        print("✓ Forward pass successful with custom TREAD settings")
        print(f"  - Output shape: {output.shape}")
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 6: Verify token routing logic
    print("\n[Test 6] Testing TokenRouter directly...")
    try:
        from model import TokenRouter
        
        router = TokenRouter(
            start_layer=2,
            end_layer=8,
            drop_percent=50,
            total_layers=12
        )
        
        # Create dummy input with registers + patches
        batch_size = 2
        num_cls_tokens = 16  # registers
        num_patches = (32 // 2) ** 2  # = 256
        hidden_dim = 256
        
        x = torch.randn(batch_size, num_cls_tokens + num_patches, hidden_dim)
        patch_shape = (16, 16)
        
        # Test routing
        route_state = router.route_tokens_from_start(x, patch_shape, num_cls_tokens)
        print(f"✓ Token routing extraction successful")
        print(f"  - Total tokens: {x.shape[1]}")
        print(f"  - Class tokens: {num_cls_tokens}")
        print(f"  - Patch tokens: {num_patches}")
        print(f"  - Direct tokens: {route_state['x_direct'].shape[1]}")
        print(f"  - Routed tokens: {route_state['x_routed'].shape[1]}")
        print(f"  - Selection rate: {router.selection_rate*100:.1f}%")
        
        # Test reintroduction
        x_reintroduced = router.reintroduce_tokens_at_end(
            route_state['x_direct'],
            route_state['x_routed'],
            route_state
        )
        print(f"✓ Token reintroduction successful")
        print(f"  - Reintroduced shape: {x_reintroduced.shape}")
        assert x_reintroduced.shape == x.shape, \
            f"Reintroduced shape {x_reintroduced.shape} != original shape {x.shape}"
        print("✓ Shape consistency verified")
        
    except Exception as e:
        print(f"✗ TokenRouter test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    print("✓ ALL TESTS PASSED!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = test_tread_basic()
    sys.exit(0 if success else 1)
