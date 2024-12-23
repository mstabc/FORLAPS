# Reinforcement Learning Framework

This repository contains a Python program for implementing reinforcement learning (RL) with or without data augmentation and for evaluating RL performance using test datasets. It supports offline RL, augmented RL, and evaluation workflows.

---

## **Project Structure**

- **`main.py`**: Main script to run the RL pipeline, including data preprocessing, RL training, augmentation, and evaluation.
- **`config.py`**: Configuration file with constants like data directory paths and RL parameters (`NUM_COP`, `ALPHA`, `GAMMA`).
- **`offline_rl.py`**: Contains the implementation of offline RL logic.
- **`data_augmentation_processor.py`**: Handles data augmentation logic.
- **`rl_evaluation.py`**: Provides functions for evaluating RL models using test data and Parquet files.

## **Usage**

Run the script with one of the three available methods:

- **Offline RL**: Train the RL model without data augmentation.
- **Augmented RL**: Train the RL model using augmented data.
- **Evaluation**: Evaluate trained models using test data.

---

### **Command Syntax**

`python main.py METHOD [OPTIONS]`

## **Parameters**

- **`METHOD`**: Specifies the operation mode (`offline`, `augmented`, or `evaluation`).
- **`--num_cop`**: (Optional) Number of data augmentation copies (default: 50).
- **`--alpha`**: (Optional) Alpha value for Q-learning (default: 0.2).
- **`--gamma`**: (Optional) Gamma value for Q-learning (default: 0.8).
- **`--data_dir`**: (Optional) Path to the directory containing input CSV files.
- **`--parquet_dir`**: (Optional) Path to the directory containing Parquet files for evaluation.

---

## **Examples**

### **Offline RL**
Train the RL model without data augmentation:

`python main.py offline --data_dir ./data --alpha 0.1 --gamma 0.9`

### **Augmented RL**
Train the RL model with augmented data:

`python main.py augmented --data_dir ./data --num_cop 100 --alpha 0.2 --gamma 0.8`


### **Evaluation**
Evaluate trained models using test datasets:
`python main.py evaluation --data_dir ./test_data --parquet_dir ./evaluation_augmented`


## **Features**

### **Reinforcement Learning**
- Supports offline RL training.
- Augmented RL for enhanced performance using synthetic data.

### **Data Augmentation**
- Generates synthetic data copies to improve RL model robustness.

### **Evaluation**
- Tests RL models using real-world test datasets.
- Outputs evaluation results in CSV format.

# LSTM and KNN Model Training and Evaluation

This project involves training a Multi-Task LSTM model and a KNN model to perform classification tasks on business process data. The LSTM model predicts the next steps and outcomes in the process, while the KNN model is used to identify similar traces based on input sequences. The main file orchestrates data preprocessing, model training, and evaluation, logging important steps and saving results for further analysis.

## **Features**

### **LSTM Model**
- Trains a Multi-Task LSTM model to predict activity and outcomes based on process data.
- Uses Cross-Entropy loss and Adam optimizer.
- Saves training and validation loss history for performance tracking.
  
### **KNN Model**
- Trains a KNN model to find similar process trace suffixes.
- Identifies alternatives based on similarity to input traces.
  
### **Data Preprocessing**
- Loads and preprocesses training data.
- Prepares traces, labels, and outcomes for the LSTM model.
  
### **Evaluation**
- Evaluates the trained models using test datasets.
- Saves results as Parquet files for easy access and analysis




## 1. Running the Script
You can run the script directly using Python, with optional arguments for configuring the model training parameters such as the number of epochs, learning rate, batch size, and hidden dimension. If no arguments are provided, the script will use default values.

### Command:

`python main.py --epochs <epochs> --lr <learning_rate> --batch_size <batch_size> --hidden_dim <hidden_dim>`

### Arguments:

- `--epochs`: Number of epochs for training the LSTM model (default: 100)
- `--lr`: Learning rate for the optimizer (default: 0.0001)
- `--batch_size`: Batch size for loading the training data (default: 64)
- `--hidden_dim`: The number of hidden dimensions in the LSTM model (default: 128)

## Example:

To run the script with custom parameters:
`python main.py --epochs 200 --lr 0.001 --batch_size 32 --hidden_dim 256`


## 2. File Processing

For each file in the `evaluation_dataset`, the script performs the following:

- Loads the dataset.
- Prepares the data for model training.
- Trains the Multi-Task LSTM model.
- Trains the KNN model.
- Saves evaluation results as Parquet files.

## 3. Evaluation

After training, the script evaluates the models on the test dataset and stores results in the `./lstm` directory.
