import pandas as pd
from sklearn.preprocessing import LabelEncoder
import logging
import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import os

logger = logging.getLogger(__name__)

def prepare_data(df):
    logger.info("Preparing data for LSTM model...")
    case_encoder = LabelEncoder()
    df['concept:name'] = case_encoder.fit_transform(df['concept:name'])

    traces, labels, outcomes = [], [], []
    grouped = df.groupby('case:concept:name')
    for _, group in grouped:
        activity_sequence = group['concept:name'].tolist()
        outcome_sequence = group['outcome'].tolist()
        for i in range(1, len(activity_sequence)):
            traces.append(activity_sequence[:i])
            labels.append(activity_sequence[i])
            outcomes.append(outcome_sequence[i])
    
    logger.info(f"Prepared {len(traces)} traces, {len(labels)} labels, and {len(outcomes)} outcomes.")
    return traces, labels, outcomes

def prepare_knn_data(df):
    trace_data = []
    grouped = df.groupby('case:concept:name')
    
    for _, group in grouped:
        group = group.sort_values(by='time:timestamp')
        trace = group['concept:name'].apply(lambda x: ord(str(x)[0]) % 32).tolist()
        outcomes = group['outcome'].tolist()

        for i in range(1, len(trace)):
            prefix = trace[:i]
            suffix = trace[i:]
            outcome = outcomes[i]
            
            if outcome == 0:
                trace_data.append((prefix, suffix))

    return trace_data

def collate_fn(batch):
    traces, labels, outcomes = zip(*batch)
    trace_lengths = torch.tensor([len(trace) for trace in traces])
    padded_traces = pad_sequence(traces, batch_first=True, padding_value=0).to('cuda')
    labels = torch.stack(labels).to('cuda')
    outcomes = torch.stack(outcomes).to('cuda')
    return padded_traces, labels, outcomes, trace_lengths