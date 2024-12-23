import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from utils import plot_loss, save_checkpoint
import logging
import os
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
class ProcessDataset(Dataset):
    def __init__(self, traces, labels, outcomes):
        self.traces = traces
        self.labels = labels
        self.outcomes = outcomes

    def __len__(self):
        return len(self.traces)

    def __getitem__(self, idx):
        trace = torch.tensor(self.traces[idx], dtype=torch.long).to('cuda')
        label = torch.tensor(self.labels[idx], dtype=torch.long).to('cuda')
        outcome = torch.tensor(self.outcomes[idx], dtype=torch.long).to('cuda')
        return trace, label, outcome



class MultiTaskLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, activity_classes, outcome_classes):
        super(MultiTaskLSTM, self).__init__()
        
        self.embedding = nn.Embedding(activity_classes, input_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.activity_fc = nn.Linear(hidden_dim, activity_classes)
        self.outcome_fc = nn.Linear(hidden_dim, outcome_classes)

    def forward(self, x, lengths):
        x = self.embedding(x)
        lengths = torch.tensor(lengths) if not isinstance(lengths, torch.Tensor) else lengths
        sorted_lengths, sorted_indices = torch.sort(lengths, descending=True)
        
        x = x[sorted_indices]
        packed_input = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        
        packed_output, (hidden, _) = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        hidden = hidden[-1]
        activity_pred = self.activity_fc(hidden)
        outcome_pred = self.outcome_fc(hidden)
        return activity_pred, outcome_pred

def train_lstm_model(model, train_loader, val_loader, criterion, optimizer, file_name="model", checkpoint_dir="checkpoints", model_dir="models", epochs=50, hidden_dim= None, activity_classes=None, outcome_classes=None):
    logger.info("Training the LSTM model...")
    model.train()
    train_loss_history, val_loss_history = [], []

    for epoch in range(epochs):
        model.train()
        total_train_loss, total_activity_loss, total_outcome_loss = 0, 0, 0
        scaler = torch.cuda.amp.GradScaler()
        for traces, labels, outcomes, lengths in train_loader:
            traces, labels, outcomes = traces.to('cuda'), labels.to('cuda'), outcomes.to('cuda')
            with torch.cuda.amp.autocast():
                activity_pred, outcome_pred = model(traces, lengths)
                activity_loss = criterion(activity_pred, labels)
                outcome_loss = criterion(outcome_pred, outcomes)
                loss = activity_loss + outcome_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            total_train_loss += loss.item()
            total_activity_loss += activity_loss.item()
            total_outcome_loss += outcome_loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_activity_loss = total_activity_loss / len(train_loader)
        avg_outcome_loss = total_outcome_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)

        if (epoch + 1) % 20 == 0: 
            checkpoint_path = os.path.join(directory, f"{file_name}_epoch{epoch+1}.pth")
            save_checkpoint(model, optimizer, epoch+1, avg_train_loss, checkpoint_path)

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for traces, labels, outcomes, lengths in val_loader:
                activity_pred, outcome_pred = model(traces, lengths)
                activity_loss = criterion(activity_pred, labels)
                outcome_loss = criterion(outcome_pred, outcomes)
                val_loss = activity_loss + outcome_loss
                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_loss_history.append(avg_val_loss)
        if (epoch + 1) % 1 == 0: 
            logger.info(f'Epoch {epoch+1}/{epochs}, Validation Loss: {avg_val_loss:.4f}')
    plot_dir = "lstm_plots"
   
    plot_file_name = f"{file_name}_train_val_loss.png"
    plot_path = os.path.join(plot_dir, plot_file_name)
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
    logger.info(f"Training and validation loss plot saved as {plot_path}")
    input_dim = 1
    if hidden_dim is None:
        hidden_dim = model.lstm.hidden_size
    directory = "lstm"
    model_path = os.path.join(directory, f"{file_name}_final_model.pth")
    torch.save(model.state_dict(), model_path)
   
    return train_loss_history, val_loss_history