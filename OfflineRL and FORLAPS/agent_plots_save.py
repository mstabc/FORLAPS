import os
import pandas as pd
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AgentPlotsSave:
    
    def save_plot(self, dataset_name, avg_Q_history, minvar_Q_history, maxvar_Q_history, output_dir):
        plot_file_path = os.path.join(output_dir, f"{dataset_name}_Q_history_plot.png")
        
        plt.figure(figsize=(10, 6))
        plt.plot(avg_Q_history, color='#d53e4f', label='Average Q-value')
        plt.plot(minvar_Q_history, color='#abdda4', alpha=0.1, label='Min Q-value')
        plt.plot(maxvar_Q_history, color='#abdda4', alpha=0.1, label='Max Q-value')

        plt.fill_between(range(len(avg_Q_history)), minvar_Q_history, maxvar_Q_history, color='#abdda4', alpha=0.4)

        plt.xlabel('Timestep')
        plt.ylabel('Q-value')
        plt.title(f'Q-value History for {dataset_name}')
        plt.legend()

        plt.savefig(plot_file_path)
        logging.info(f"Saved Q-history plot to {plot_file_path}")
        plt.close()

    def save_best_actions(self, best_actions, dataset_name, output_dir):

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        parquet_file_path = os.path.join(output_dir, f"{dataset_name}_best_actions.parquet")
        df_best_actions = pd.DataFrame(
            list(best_actions.items()), columns=['State', 'Best_Action']
        )
        df_best_actions.to_parquet(parquet_file_path, index=False)
        logging.info(f"Saved best actions to Parquet file: {parquet_file_path}")

        