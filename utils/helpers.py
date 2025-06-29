import torch
import os

def save_checkpoint(model, optimizer, epoch, loss, path):
    """
    Saves model and optimizer state.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss}, path)

def load_checkpoint(path, model, optimizer=None):
    """
    Loads model and optimizer state from a checkpoint.
    """
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint.get('epoch', 0), checkpoint.get('loss', None)

def count_parameters(model):
    """
    Returns the total number of trainable parameters in the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_model_summary(model):
    """
    Prints the number of trainable parameters.
    """
    print(f"Total trainable parameters: {count_parameters(model):,}")
