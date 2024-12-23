import matplotlib.pyplot as plt
import os

def plot_loss(train_loss_history, val_loss_history, plot_dir, file_name, epochs):
    plot_path = os.path.join(plot_dir, file_name)
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.figure()
    plt.plot(range(1, epochs + 1), train_loss_history,  color="b", label="Training Loss")
    plt.plot(range(1, epochs + 1), val_loss_history,  color="r", label="Validation Loss")
    plt.title('LSTM Training and Validation Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.savefig(plot_path)
    plt.show()

def save_checkpoint(model, optimizer, epoch, loss, file_path):
    """Save a checkpoint for the model."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, file_path)
