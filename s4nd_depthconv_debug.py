import torch
from src.models.sequence.modules.s4nd_depthconv import S4ND



if __name__ == "__main__":
    # Create a small test instance
    model = S4ND(
        d_model=32,
        d_state=8,
        l_max=128,
        channels=3,
        dim=2,
        transposed=True
    )

    # Test forward pass with random input
    batch_size = 2
    h = 64
    w = 64
    x = torch.randn(batch_size, 3, h, w)  # B C H W format
    
    print("Testing forward pass...")
    # try:
    out = model(x)
    print(f"Forward pass successful. Output shape: {out.shape}")
    # except Exception as e:
        # print(f"Forward pass failed with error: {e}")

    # Test rank adaptation
    print("\nTesting rank adaptation...")
    print(f"Base rank: {model.base_rank}")
    print(f"Adaptive rank: {model.adaptive_rank}")
    
    # Test regularization
    print("\nTesting regularization...")
    reg_loss = model.get_regularization_loss()
    print(f"Regularization loss: {reg_loss.item()}")

    # Test rank structure analysis
    print("\nAnalyzing rank structure...")
    rank_info = model.analyze_rank_structure()
    print(f"Effective rank: {rank_info['effective_rank']}")
    print(f"Condition numbers: {rank_info['condition_numbers']}")
    
    # Test orthogonalization
    print("\nTesting projection orthogonalization...")
    model.orthogonalize_projections()
    print("Orthogonalization complete")

    print("\nAll tests completed successfully!")

