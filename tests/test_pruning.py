import pytest
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from src.pruning import apply_weight_pruning

@pytest.fixture
def simple_model():
    """Fixture to create a simple linear model for testing."""
    model = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 3)
    )
    return model

def test_apply_weight_pruning(simple_model):
    """Tests the apply_weight_pruning function against PyTorch's pruning."""
    # Clone the model for comparison
    torch_model = simple_model
    custom_model = simple_model

    # Apply PyTorch's pruning
    for _, module in torch_model.named_modules():
        if isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=0.5)

    # Apply custom weight pruning
    pruned_custom_model = apply_weight_pruning(custom_model, nn.Linear, pruning_percentage=0.5)

    # Compare pruned weights
    for custom_module, torch_module in zip(pruned_custom_model.modules(), torch_model.modules()):
        if isinstance(custom_module, nn.Linear):
            custom_pruned_weights = custom_module.weight.data
            torch_pruned_weights = torch_module.weight.data
            # Ensure the same weights are pruned
            assert torch.allclose(custom_pruned_weights, torch_pruned_weights), \
                "Custom weight pruning does not match PyTorch's pruning"

