from data_handler_processor import DataHandlerProcessor
from agent_plots_save import AgentPlotsSave
import numpy as np
import logging
import os
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class ReinforcementLearningProcessor:
    def __init__(self, directory, method, plot_directory='plots', alpha=0.2, gamma=0.8):
        self.alpha = alpha
        self.gamma = gamma
        self.q_histories = {}
        self.best_actions = {}
        self.method = method
        self.data_handler = DataHandlerProcessor(directory)
        self.agent_plots = AgentPlotsSave()

        self.output_dir = "./evaluation_augmented" if method == "augmented" else "./evaluation_not_augmented"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        
    def run_rl_algorithm(self, df, dataset_name, all_len):
        episodes = self.data_handler.process_episode(df, dataset_name)
        actions = df['concept:name'].unique()
        Q = {}
        Q_history = []
        avg_Q_history = []
        minvar_Q_history = []
        maxvar_Q_history = []

        def get_q_value(Q, state, action):
            if (state, action) not in Q:
                Q[(state, action)] = 0
            return Q[(state, action)]

        def update_q_table(Q, state, action, reward, next_state):
            current_q = get_q_value(Q, state, action)
            max_future_q = max(get_q_value(Q, next_state, a) for a in actions)
            new_q = current_q + self.alpha * (reward + self.gamma * max_future_q - current_q)
            Q[(state, action)] = new_q

        window_size = int(all_len / 5)
        for episode_id, episode in episodes.items():
            state = ()
            label = episode[1]
            for i in range(len(episode[0])):
                action = episode[0][i]
                next_state = state + (action,)
                reward = -0.01 * i if label == 1 else 0.3
                update_q_table(Q, state, action, reward, next_state)
                Q_history.append(Q[(state, action)])
                state = next_state

            avg_Q_history.append(np.mean(Q_history[-window_size:]))
            minvar_Q_history.append(avg_Q_history[-1] - np.std(Q_history[-window_size:]))
            maxvar_Q_history.append(avg_Q_history[-1] + np.std(Q_history[-window_size:]))

        self.q_histories[dataset_name] = (avg_Q_history, minvar_Q_history, maxvar_Q_history)
        self.agent_plots.save_plot(dataset_name, avg_Q_history, minvar_Q_history, maxvar_Q_history, self.output_dir)


        grouped_Q = self.data_handler.group_by_state(Q)
        self.best_actions[dataset_name] = self.data_handler.find_best_actions(grouped_Q)
        best_action_file = os.path.join(self.output_dir, f"{dataset_name}_best_actions.parquet")
        self.data_handler.save_to_parquet(self.best_actions[dataset_name], best_action_file)
        logging.info(f"Best actions saved to {best_action_file}")
        plot_file = os.path.join(self.output_dir, f"{dataset_name}_q_value_plot.png")
        self.agent_plots.save_best_actions(self.best_actions[dataset_name], dataset_name, self.output_dir)

    def process_all_datasets(self):
        self.data_handler.process_all_datasets(self)
