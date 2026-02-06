import torch
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.cbm.concept_head import ConceptPredictor

def test_concept_predictor():
    print("Testing ConceptPredictor...")
    
    # Initialize model
    model = ConceptPredictor(backbone_name='resnet18', pretrained=False)
    model.eval()
    
    # Create dummy input: Batch size 2, 3 channels, 224x224
    dummy_input = torch.randn(2, 3, 224, 224)
    
    # Forward pass
    with torch.no_grad():
        output = model(dummy_input)
        
    print(f"Output shape: {output.shape}")
    
    # Check shape
    if output.shape == (2, 4):
        print("Shape is correct: (Batch, 4)")
    else:
        print(f"Shape mismatch: Expected (2, 4), got {output.shape}")
        
    # Check range (since we use random weights, the pre-clamp values could be anything)
    # But post-clamp must be 0-5
    min_val = output.min().item()
    max_val = output.max().item()
    
    print(f"Output Range: [{min_val:.4f}, {max_val:.4f}]")
    
    if min_val >= 0 and max_val <= 5:
        print("Output is strictly within [0, 5]")
    else:
        print("Output values valid range [0, 5]")
        
    print("Test Complete.")

if __name__ == "__main__":
    test_concept_predictor()
