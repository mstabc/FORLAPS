import pandas as pd
import random
import logging
from config import DIRECTORY, NUM_COP

class DataAugmentationProcessor:
    def __init__(self, num_copies = NUM_COP):
        self.num_copies = num_copies
        logging.info("Initialized DataAugmentationProcessor with num_copies: %d", num_copies)

    def augment_data(self, df):
        """
        Augments the data by creating copies and deleting 20% of the rows randomly.
        """
        df_new = pd.DataFrame()

        for i in range(1, self.num_copies + 1):
            df_duplicate = df.copy()
            df_duplicate['case:concept:name'] = df_duplicate['case:concept:name'].astype(str) + f'_{i}'
            df_new = pd.concat([df_new, df_duplicate], ignore_index=True)
        num_rows_to_delete = round(len(df_new) / 5)
        rows_to_delete = random.sample(list(df_new.index), num_rows_to_delete)
        df_augmented = df_new.drop(rows_to_delete).reset_index(drop=True)
        
        logging.info(f'Augmented data: created {self.num_copies} copies with 20% random deletions')
        return df_augmented
