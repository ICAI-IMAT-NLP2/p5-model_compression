import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from src.distillation import distill_model, compute_distillation_loss

@pytest.fixture
def teacher_student_models():
    """Fixture to create a simple teacher and student model for testing."""
    teacher_model = nn.Sequential(
        nn.Linear(4, 16),
        nn.ReLU(),
        nn.Linear(16, 3)
    )
    student_model = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 3)
    )
    return teacher_model, student_model

@pytest.fixture
def sample_data():
    """Fixture to create sample data for testing."""
    inputs = torch.rand(20, 4)  # 20 samples, 4 features
    labels = torch.randint(0, 3, (20,))  # 20 samples, 3 classes
    dataset = TensorDataset(inputs, labels)
    return DataLoader(dataset, batch_size=5)

def test_compute_distillation_loss(teacher_student_models):
    """Tests the compute_distillation_loss function with different temperatures."""
    student_logits = torch.tensor([[ 0.1140,  0.2251,  0.2423], [ 0.0956,  0.1710,  0.2188], [-0.0062,  0.1275,  0.2613], [-0.0086,  0.1290,  0.2029], [ 0.0856,  0.2050,  0.2436], [ 0.0908,  0.1493,  0.2196], [-0.0038,  0.0634,  0.1610], [-0.0463,  0.0709,  0.2235], [ 0.0841,  0.2026,  0.2446], [ 0.0058,  0.1287,  0.1599]])
    teacher_logits = torch.tensor([[-0.1310, -0.1369, -0.0379], [-0.0858, -0.1036, -0.0735], [-0.0460, -0.1537, -0.0254], [-0.0010, -0.1187, -0.0422], [-0.0653, -0.1515, -0.0490], [-0.0756, -0.1017, -0.0737], [ 0.0883, -0.0757, -0.0951], [ 0.0775, -0.1400, -0.0928], [-0.1119, -0.1619, -0.0534], [ 0.0368, -0.0679, -0.0609]])
    
    # Test distillation loss at different temperatures
    loss_1 = compute_distillation_loss(student_logits, teacher_logits, temperature=1.0)
    assert loss_1.item() > 0, "Distillation loss at temperature=1.0 should be positive"
    assert loss_1.round(decimals=4) == 0.0059, "Incorrect distillation loss at temperature=1.0"

    loss_5 = compute_distillation_loss(student_logits, teacher_logits, temperature=5.0)
    assert loss_5.item() > 0, "Distillation loss at temperature=5.0 should be positive"
    assert loss_5.round(decimals=4) == 0.0058, "Incorrect distillation loss at temperature=5.0"

def test_distill_model(teacher_student_models, sample_data):
    """Tests the distill_model function to ensure the student model is trained via distillation
       and that the loss decreases over time.

    Args:
        teacher_student_models: Fixture providing teacher and student models.
        sample_data: Fixture providing sample data for training.
    """
    teacher_model, student_model = teacher_student_models
    initial_student_params = [param.clone() for param in student_model.parameters()]

    # Run distillation training
    trained_student, epoch_losses = distill_model(
        teacher_model=teacher_model,
        student_model=student_model,
        train_data=sample_data,
        temperature=2.0,
        alpha=0.5,
        num_epochs=5,
        learning_rate=0.01
    )

    # Ensure that the student model parameters have been updated
    for initial, trained in zip(initial_student_params, trained_student.parameters()):
        assert not torch.equal(initial, trained), "Student model parameters should change after training"

    # Check that the loss decreases over epochs
    assert epoch_losses[0] > epoch_losses[-1], "Final loss should be lower than initial loss, indicating learning"
