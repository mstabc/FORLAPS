import argparse
import logging
import pandas as pd
from config import DIRECTORY, NUM_COP, ALPHA, GAMMA
from data_augmentation_processor import DataAugmentationProcessor
from offline_rl import ReinforcementLearningProcessor
from rl_evaluation import process_test_data_with_parquet
import os

def evaluate(data_dir, parquet_dir):
    """Run evaluation on the specified dataset."""
    # parquet_dir = "./evaluation_augmented" if method == "augmented" else "./evaluation_not_augmented"
    logging.info(f"Using parquet directory: {parquet_dir}")
    test_dir = data_dir
    results = []

    for file in os.listdir(test_dir):
        if file.endswith("_test.csv"):
            file_name = file.replace("_test.csv", "")
            parquet_file = os.path.join(parquet_dir, f"{file_name}_best_actions.parquet")
            logging.info("Processing file: %s", file_name)
            if not os.path.exists(parquet_file):
                logging.warning(f"Missing Parquet file for {file_name}. Skipping.")
                continue

            test_csv_path = os.path.join(test_dir, file)
            test_data = pd.read_csv(test_csv_path)
            parquet_data = pd.read_parquet(parquet_file)

            file_results = process_test_data_with_parquet(test_data, parquet_data)
            results_df = pd.DataFrame(file_results)
            output_file = os.path.join(parquet_dir, f"{file_name}_results.csv")
            results_df.to_csv(output_file, index=False)

            logging.info(f"Results saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Run Reinforcement Learning with or without data augmentation.")
    parser.add_argument('method', choices=['offline', 'augmented', 'evaluation_offline', 'evaluation_augmented'], 
                        help="Specify the RL method: 'offline', 'augmented', 'evaluation_offline', or 'evaluation_augmented'.")
    parser.add_argument('--num_cop', type=int, default=NUM_COP, 
                        help="Number of copies for data augmentation (default is 50).")
    parser.add_argument('--alpha', type=float, default=ALPHA, 
                        help="Alpha value for Q-learning (default is 0.2).")
    parser.add_argument('--gamma', type=float, default=GAMMA, 
                        help="Gamma value for Q-learning (default is 0.8).")
    parser.add_argument('--data_dir', type=str, default=DIRECTORY, 
                        help="Directory containing the CSV files.")
    parser.add_argument('--parquet_dir', type=str, default='./evaluation_augmented', 
                        help="Directory containing Parquet files for evaluation.")
    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
    
    if args.method in ['offline', 'augmented']:
        processor = ReinforcementLearningProcessor(
            directory=args.data_dir, 
            method=args.method,
            alpha=args.alpha, 
            gamma=args.gamma
        )
        files = [f for f in os.listdir(args.data_dir) if f.endswith('_train.csv')]
        
        for file in files:
            file_path = os.path.join(args.data_dir, file)
            logging.info(f"Processing file: {file_path}")
            df = pd.read_csv(file_path)
            
            if args.method == 'offline':
                logging.info("Running offline RL on the dataset.")
                processor.run_rl_algorithm(df, dataset_name=file.split('_train')[0], all_len=len(df))
            
            elif args.method == 'augmented':
                logging.info("Running augmented RL on the dataset.")
                augmentor = DataAugmentationProcessor(num_copies=args.num_cop)
                df_augmented = augmentor.augment_data(df)
                processor.run_rl_algorithm(df_augmented, dataset_name=file.split('_train')[0], all_len=len(df_augmented))
    
    elif args.method == 'evaluation_augmented':
        logging.info("Running evaluation on the augmented dataset.")
        evaluate(data_dir=args.data_dir, parquet_dir='./evaluation_augmented')

    elif args.method == 'evaluation_offline':
        logging.info("Running evaluation on the offline dataset.")
        evaluate(data_dir=args.data_dir, parquet_dir='./evaluation_not_augmented')

if __name__ == '__main__':
    main()