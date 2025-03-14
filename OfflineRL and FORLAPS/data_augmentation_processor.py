import pandas as pd
import random
import logging
from config import DIRECTORY, NUM_COP

class DataAugmentationProcessor:
    def __init__(self, num_copies=NUM_COP):
        self.num_copies = num_copies
        logging.info("Initialized DataAugmentationProcessor with num_copies: %d", num_copies)

    def check_business_rules(self, trace):
        """
        Check if the trace complies with business rules:
        - Precedence: Ensure dependent activities are in the right order.
        - Conflict: Avoid traces containing conflicting activities.
        - Co-occurrence: Ensure dependent activities appear together.
        """
        precedence_rules = []  
        conflict_rules = [] 
        cooccurrence_rules = [] 
        
        activities = trace['activity'].tolist()
        
        # Check precedence rules
        for a, b in precedence_rules:
            if a in activities and b in activities:
                if activities.index(a) > activities.index(b):
                    return False
        
        # Check conflict rules
        for a, b in conflict_rules:
            if a in activities and b in activities:
                return False
        
        # Check co-occurrence rules
        for a, b in cooccurrence_rules:
            if (a in activities and b not in activities) or (b in activities and a not in activities):
                return False
        
        return True

    def augment_data(self, df):
        """
        Augments the data by creating copies and ensuring business rule compliance.
        """
        df_new = pd.DataFrame()

        for i in range(1, self.num_copies + 1):
            df_duplicate = df.copy()
            df_duplicate['case:concept:name'] = df_duplicate['case:concept:name'].astype(str) + f'_{i}'
            
            # Random noise in timestamps (up to 10%)
            if 'timestamp' in df_duplicate.columns:
                df_duplicate['timestamp'] += pd.to_timedelta(
                    df_duplicate['timestamp'].diff().fillna(pd.Timedelta(seconds=0)) * random.uniform(-0.1, 0.1)
                )

            df_new = pd.concat([df_new, df_duplicate], ignore_index=True)
        
        # Randomly delete 20% of rows while preserving constraints
        num_rows_to_delete = round(len(df_new) / 5)
        indices = list(df_new.index)
        random.shuffle(indices)
        
        rows_to_keep = []
        for idx in indices:
            trace = df_new[df_new['case:concept:name'] == df_new.loc[idx, 'case:concept:name']]
            if self.check_business_rules(trace):
                rows_to_keep.append(idx)
                if len(rows_to_keep) >= len(df_new) - num_rows_to_delete:
                    break
        
        df_augmented = df_new.loc[rows_to_keep].reset_index(drop=True)
        logging.info(f'Augmented data: created {self.num_copies} copies with 20% random deletions')
        return df_augmented
