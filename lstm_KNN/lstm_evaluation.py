import os
import torch
import pandas as pd
from jellyfish import damerau_levenshtein_distance
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from lstm_model import MultiTaskLSTM, ProcessDataset
# from lstm_model import ProcessDataset

import logging
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


    
def load_and_encode_activity(train_file, test_file):
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)

    activity_encoder = LabelEncoder()
    train_data['concept:name_encoded'] = activity_encoder.fit_transform(train_data['concept:name'])
    
    train_activities = set(train_data['concept:name'])
    test_data = test_data[test_data['concept:name'].isin(train_activities)]

    test_data['concept:name_encoded'] = activity_encoder.transform(test_data['concept:name'])

    
    train_data['concept:name1'] = train_data['concept:name_encoded'].apply(lambda x: ord(str(x)[0]) % 32)
    test_data['concept:name1'] = test_data['concept:name_encoded'].apply(lambda x: ord(str(x)[0]) % 32)

    encode_map = dict(zip(train_data['concept:name_encoded'], train_data['concept:name1']))
    decode_map = {v: k for k, v in encode_map.items()}
    return train_data, test_data, activity_encoder, encode_map, decode_map

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

def decode_task(encoded_task, decode_map, activity_encoder):
    label_encoded_task = decode_map.get(encoded_task, None)
    if label_encoded_task is None:
        return "UNKNOWN"
    return activity_encoder.inverse_transform([label_encoded_task])[0]


def process_test_data(test_data, parquet_data, lstm_model, activity_encoder, encode_map, decode_map):
    results = []
    grouped = test_data.groupby('case:concept:name')

    parquet_data['input_trace'] = parquet_data['input_trace'].astype(str)
    parquet_data['input_trace'] = (
    parquet_data['input_trace']
    .str.replace(r"'\s*'", "', '", regex=True)  # Add a comma and space between tasks
    .str.replace(r"[\[\]']", '', regex=True)   # Remove brackets and single quotes
    .str.strip()
    )                
    parquet_data['alternatives'] = parquet_data['alternatives'].astype(str)
    parquet_data['alternatives'] = (
    parquet_data['alternatives']
    .str.replace(r"'\s*'", "', '", regex=True)  # Add a comma and space between tasks
    .str.replace(r"[\[\]']", '', regex=True)   # Remove brackets and single quotes
    .str.strip()  
    # Remove leading/trailing spaces
    )  
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  
    
    
    for case_id, group in grouped:
        activity_sequence = group['concept:name'].tolist()

        for i in range(1, len(activity_sequence)):
            prefix = activity_sequence[:i]
            prefix_encoded = [activity_encoder.transform([activity])[0] for activity in prefix]
         
            suffix = activity_sequence[i:]
            prefix_tensor = torch.tensor([prefix_encoded], dtype=torch.long)
            lengths = [len(prefix)]

            lstm_model.eval()
            with torch.no_grad():
                _, outcome_prediction = lstm_model(prefix_tensor, lengths)
                predicted_outcome = torch.argmax(outcome_prediction, dim=1).item()

            suffix = activity_sequence[i:]
            suffix_str = ', '.join(suffix)

            if predicted_outcome == 1:
                alt_suffix_values = parquet_data.loc[
                    parquet_data['input_trace'] == ' '.join(map(str, prefix_encoded)),
                    'alternatives'
                ].values
                alt_suffix = alt_suffix_values[0] if len(alt_suffix_values) > 0 else ""
                alt_suffix_encoded = list(map(int, alt_suffix.split())) if alt_suffix else []
                alt_suffix_decoded = [decode_task(task, decode_map, activity_encoder) for task in alt_suffix_encoded]
                alt_suffix_decoded_str = ', '.join(alt_suffix_decoded)
                distance = damerau_levenshtein_distance(alt_suffix_decoded_str, suffix_str)

                results.append({
                    'case_id': case_id,
                    'prefix': prefix,
                    'suffix': suffix_str,
                    'predicted_outcome': predicted_outcome,
                    'alt_suffix': alt_suffix_decoded_str,
                    'distance': distance
                })

    return results


def evaluate_lstm(test_dir, model_dir):
    file_pairs = []
    for file in os.listdir(test_dir):
        if file.endswith("_test.csv"):
            base_name = file.replace("_test.csv", "")
            parquet_file = os.path.join(model_dir, f"{base_name}_train.csv_results.parquet")
            model_file = os.path.join(model_dir, f"{base_name}_train.csv_final_model.pth")
            if os.path.exists(parquet_file) and os.path.exists(model_file):
                file_pairs.append((file, parquet_file, model_file))
            else:
                logger.warning(f"Missing files for {base_name}. Skipping...")

    for test_file, parquet_file, model_file in file_pairs:
        logger.info(f"Evaluating: {test_file}")
        test_csv_path = os.path.join(test_dir, test_file)
        train_csv_path = test_csv_path.replace("_test.csv", "_train.csv")


        train_data, test_data, activity_encoder, encode_map, decode_map = load_and_encode_activity(train_csv_path, test_csv_path)
        parquet_data = pd.read_parquet(parquet_file)

        activity_classes = len(train_data['concept:name'].unique())
        outcome_classes= len(train_data['outcome'].unique())
        model_params = torch.load(model_file)
        
        lstm_model = MultiTaskLSTM(
            input_dim= 1 ,
            hidden_dim=128,
            activity_classes=activity_classes,
            outcome_classes=outcome_classes
        )
        lstm_model.load_state_dict(torch.load(model_file))


        results = process_test_data(test_data, parquet_data, lstm_model, activity_encoder, encode_map, decode_map)
        results_df = pd.DataFrame(results)
        results_df.to_csv(f"{model_dir}/{test_file}_evaluation_results.csv", index=False)
        logger.info(f"Saved results for {test_file}")
