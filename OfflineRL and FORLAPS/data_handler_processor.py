import os
import pandas as pd
import numpy as np
from collections import defaultdict
import logging
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class DataHandlerProcessor:
    def __init__(self, directory):
        self.directory = directory

    def load_datasets(self):
        files = [f for f in os.listdir(self.directory) if f.endswith('_train.csv')]
        return files

    def group_by_state(self, Q):
        grouped = defaultdict(list)
        for (state, action), q_val in Q.items():
            grouped[state].append((action, q_val))
        return grouped

    def find_best_actions(self, grouped_Q):
        return {state: max(actions, key=lambda x: x[1])[0] for state, actions in grouped_Q.items()}

    def parallel_find_best_actions(self, grouped_Q):
        with ThreadPoolExecutor() as executor:
            results = executor.map(
                lambda state_actions: (state_actions[0], max(state_actions[1], key=lambda x: x[1])[0]),
                grouped_Q.items()
            )
        return dict(results)

    def process_episode(self, df, file_name):
        episodes = {}
        for case, group in df.groupby('case:concept:name'):
            actions = list(group['concept:name'])
            label = group['outcome'].iloc[0]
            episodes[case] = (actions, label)
        logging.info("Processed episodes from file: %s", file_name)
        return episodes
    def save_to_parquet(self, data, file_path):
        """Save the data to a Parquet file."""
        if isinstance(data, pd.DataFrame):
            data.to_parquet(file_path, index=False)
            logging.info(f"Data saved to Parquet at {file_path}")
        else:
            logging.error("Provided data is not a DataFrame, cannot save to Parquet.")
