import os
import glob
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from lstm_model import ProcessDataset, MultiTaskLSTM, train_lstm_model
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import argparse
from data_preprocessing import prepare_data, prepare_knn_data, collate_fn
from knn_model import train_knn_model, get_similar_suffix
from lstm_evaluation import evaluate_lstm
from utils import plot_loss, save_checkpoint


import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
def parse_args():
    parser = argparse.ArgumentParser(description="Train LSTM model for process data")
    
    parser.add_argument('--epochs', type=int, default=100, help="Number of epochs (default: 100)")
    parser.add_argument('--lr', type=float, default=0.0001, help="Learning rate (default: 0.0001)")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument('--hidden_dim', type=int, default=128, help="Hidden dimension (default: 128)")
    
    return parser.parse_args()

args = parse_args()
directory = '../evaluation_dataset'
train_files = glob.glob(os.path.join(directory, '*train*.csv'))

for file in train_files:
    logger.info(f'Processing file: {file}')
    try:
        df = pd.read_csv(file)
        logger.info(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns.")
    except Exception as e:
        logger.error(f"Error loading file {file}: {e}")
        continue
    
    
    traces, labels, outcomes = prepare_data(df)
    
    dataset = ProcessDataset(traces, labels, outcomes)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    input_dim = 1
    hidden_dim = args.hidden_dim
    activity_classes = len(df['concept:name'].unique())
    outcome_classes = len(df['outcome'].unique())

    lstm_model = MultiTaskLSTM(input_dim, hidden_dim, activity_classes, outcome_classes).to('cuda')
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(lstm_model.parameters(), lr=args.lr)

    base_file_name = file.split("/")[-1].split("_Train")[0]
    train_loss_history, val_loss_history = train_lstm_model(
        model=lstm_model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        file_name=base_file_name, 
        epochs=args.epochs,
        hidden_dim=args.hidden_dim
    )
    
    logger.info(f"Training and validation for file {file} completed.")

    # Train KNN model
    knn_model, trace_data = train_knn_model(df)
    results = pd.DataFrame({
        'input_trace': [trace[0] for trace in trace_data],  
        'alternatives': [
            [suffix for alternative in get_similar_suffix(trace[0], knn_model, trace_data) for suffix in alternative]
            for trace in trace_data 
        ]
    })

    directory = './lstm'
    test_directory = "../evaluation_dataset"
    model_directory = "./lstm/evaluation_dataset/"
    os.makedirs(directory, exist_ok=True)

    parquet_file_name = os.path.join(directory, f"{base_file_name}_results.parquet")
    results.to_parquet(parquet_file_name, engine='pyarrow')
    evaluate_lstm(test_directory, model_directory)