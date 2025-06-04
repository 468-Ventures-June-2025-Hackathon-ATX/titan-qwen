#!/usr/bin/env python3
"""
Test script to verify TITANS installation and basic functionality
"""

import torch
from titans_pytorch import NeuralMemory, MemoryAsContextTransformer

def test_neural_memory():
    """Test basic neural memory functionality"""
    print("Testing NeuralMemory...")
    
    # Create a neural memory instance
    mem = NeuralMemory(
        dim=384,
        chunk_size=64
    )
    
    # Test with a sample sequence
    seq = torch.randn(2, 1024, 384)
    retrieved, mem_state = mem(seq)
    
    print(f"Input shape: {seq.shape}")
    print(f"Output shape: {retrieved.shape}")
    print(f"Memory state type: {type(mem_state)}")
    
    assert seq.shape == retrieved.shape, "Input and output shapes should match"
    print("✓ NeuralMemory test passed!")
    return True

def test_mac_transformer():
    """Test Memory-as-Context transformer"""
    print("\nTesting MemoryAsContextTransformer...")
    
    transformer = MemoryAsContextTransformer(
        num_tokens=256,
        dim=256,
        depth=2,
        segment_len=128,
        num_persist_mem_tokens=4,
        num_longterm_mem_tokens=16,
    )
    
    # Test forward pass
    token_ids = torch.randint(0, 256, (1, 1023))
    
    # Test loss computation
    loss = transformer(token_ids, return_loss=True)
    print(f"Loss: {loss.item()}")
    
    # Test inference
    with torch.no_grad():
        logits = transformer(token_ids[:, :100])
        print(f"Logits shape: {logits.shape}")
    
    print("✓ MemoryAsContextTransformer test passed!")
    return True

def main():
    """Run all tests"""
    print("TITANS PyTorch Installation Test")
    print("=" * 40)
    
    try:
        test_neural_memory()
        test_mac_transformer()
        print("\n🎉 All tests passed! TITANS is working correctly.")
        return True
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main()
