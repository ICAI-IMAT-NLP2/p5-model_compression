import torch
import torch.nn as nn

def apply_weight_pruning(model: nn.Module, layer_type: type, pruning_percentage: float) -> nn.Module:
    """Applies unstructured weight pruning by zeroing out the smallest magnitude weights.

    Args:
        model: The neural network model to be pruned.
        layer_type: The type of layer to apply structured pruning (e.g., nn.Linear).
        pruning_percentage: The percentage of weights to prune (set to zero) based on magnitude.

    Returns:
        The model with pruned weights.
    """
    # Loop through each module in the model
    for name, module in model.named_modules():
        if isinstance(module, layer_type):

            # TODO: Get the absolute values of the weights and flatten them to calculate the pruning threshold
            weights: torch.Tensor = None

            # TODO: Calculate the pruning threshold based on the pruning_percentage
            threshold: float = None

            # TODO: Create a mask for weights above the threshold and apply it to prune the model
            mask: torch.Tensor = None
            module.weight.data.mul_(mask)

    return model
