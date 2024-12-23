import numpy as np
from sklearn.neighbors import NearestNeighbors
from data_preprocessing import prepare_knn_data
import os

import logging
logger = logging.getLogger(__name__)


def train_knn_model(df):
    trace_data = prepare_knn_data(df)
    X = np.array([np.mean(prefix) for prefix, _ in trace_data]).reshape(-1, 1)
    knn_model = NearestNeighbors(n_neighbors=1, algorithm='auto')
    knn_model.fit(X)
    return knn_model, trace_data

def get_similar_suffix(input_prefix, knn_model, trace_data):
    if isinstance(input_prefix, (int, float)):
        logger.warning(f"Expected a list/sequence, but got scalar: {input_prefix}")
        return []

    input_encoded = [ord(str(x)[0]) % 32 for x in input_prefix]
    input_feature = np.array([[np.mean(input_encoded)]])
    distances, indices = knn_model.kneighbors(input_feature)
    similar_traces = trace_data[indices[0][0]] 
    return similar_traces
