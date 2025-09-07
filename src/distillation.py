import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

def distill_model(
    teacher_model: nn.Module,
    student_model: nn.Module,
    train_data: DataLoader,
    temperature: float,
    alpha: float,
    num_epochs: int,
    learning_rate: float
) -> tuple[nn.Module, list[float]]:
    """Trains the student model to mimic the teacher model using knowledge distillation.

    Args:
        teacher_model: The pre-trained, larger model serving as the teacher.
        student_model: The smaller model to be trained.
        train_data: The DataLoader for the training dataset.
        temperature: The temperature parameter for softening probabilities.
        alpha: Weighting factor between the distillation loss and the original loss.
        num_epochs: Number of training epochs.
        learning_rate: Learning rate for the optimizer.

    Returns:
        The trained student model and a list of average losses per epoch.
    """
    # TODO: Set teacher to evaluation mode
    
    # TODO: Set student to training mode
    
    # TODO: Initialize Adam optimizer with student model parameters and learning rate
    optimizer = None

    # TODO: Define the criterion for cross-entropy loss
    criterion = None

    # Initialize list to track average loss for each epoch
    epoch_losses: list[float] = []

    for epoch in range(num_epochs):
        # TODO: Ensure the student model is in training mode
        student_model.train()

        # Initialize the epoch's loss to zero
        epoch_loss = 0.0

        for inputs, labels in train_data:
            # TODO: Zero out the gradients from the previous batch
            
            # TODO: Perform forward pass for the student model
            student_outputs: torch.Tensor = None

            # TODO: Perform forward pass for the teacher model in evaluation mode
            with torch.no_grad():
                teacher_outputs: torch.Tensor = None

            # TODO: Calculate the cross-entropy loss between student outputs and labels
            loss_ce: torch.Tensor = None

            # TODO: Calculate the distillation loss between student and teacher outputs
            loss_kl: torch.Tensor = compute_distillation_loss(None, None, None)

            # TODO: Combine cross-entropy loss and distillation loss weighted by alpha parameter
            loss: torch.Tensor = None

            # TODO: Perform backpropagation and optimization
            
            # Accumulate the batch loss into the epoch loss
            epoch_loss += loss.item()

        # Calculate the average loss for the epoch and append to the epoch_losses list
        avg_epoch_loss = epoch_loss / len(train_data)
        epoch_losses.append(avg_epoch_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_epoch_loss:.4f}")

    return student_model, epoch_losses


def compute_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float
) -> torch.Tensor:
    """Calculates the distillation loss between the student and teacher model outputs.

    Args:
        student_logits: Logits output by the student model.
        teacher_logits: Logits output by the teacher model.
        temperature: The temperature parameter for softening probabilities.

    Returns:
        The computed distillation loss.
    """
    # TODO: Apply log softmax to student logits with temperature scaling
    student_probs: torch.Tensor = None

    # TODO: Apply softmax to teacher logits with temperature scaling
    teacher_probs: torch.Tensor = None

    # TODO: Calculate KL divergence loss and scale by temperature squared
    loss: torch.Tensor = None

    return loss
