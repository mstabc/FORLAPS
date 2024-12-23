import os
import pandas as pd
from jellyfish import damerau_levenshtein_distance
import logging
import numpy as np


import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
def process_test_data_with_parquet(test_data, parquet_data):
    """
    Processes the test data using the Parquet file for best actions.
    Ensures robust handling of prefixes not found in the Parquet data.
    """
    results = []
    grouped = test_data.groupby('case:concept:name')

    parquet_data['State'] = parquet_data['State'].astype(str)
    parquet_data['State'] = (
    parquet_data['State']
    .str.replace(r"'\s*'", "', '", regex=True)
    .str.replace(r"[\[\]']", '', regex=True)   
    .str.strip()                              
)
    parquet_data['Best_Action'] = parquet_data['Best_Action'].fillna('').astype(str)

    for case_id, group in grouped:
        activity_sequence = group['concept:name'].tolist()
        outcome = group['outcome'].values[0]
        for i in range(1, len(activity_sequence)):
            prefix = activity_sequence[:i]
            suffix = activity_sequence[i:]
            
            predicted_prefix = prefix[:]
            
            for _ in range(len(suffix)):

                current_prefix_str = ', '.join(predicted_prefix)
                best_action_row = parquet_data[parquet_data['State'] == current_prefix_str]

                if best_action_row.empty:

                    break

                best_action = best_action_row['Best_Action'].values[0]

                if not best_action:
                    break

                predicted_prefix.append(best_action)

                if len(predicted_prefix) >= len(activity_sequence):
                    break
            predicted_suffix_str = ''.join(predicted_prefix[len(prefix):])
            suffix_str = ''.join(suffix)
            distance = damerau_levenshtein_distance(predicted_suffix_str, suffix_str)

            results.append({
                'case_id': case_id,
                'prefix': prefix,
                'predicted_prefix': predicted_prefix[len(prefix):],
                'suffix': suffix,
                'distance': distance,
                'outcome': outcome
            })

    print(f"Processed {len(results)} results.")
    return results