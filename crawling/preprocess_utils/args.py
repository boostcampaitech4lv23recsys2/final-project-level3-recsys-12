import pandas as pd
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/")
    parser.add_argument("--output_path", type=str, default="output/")
    parser.add_argument("--test", type=int, default=0)
    
    # for similarity
    parser.add_argument("--window_size", default=100, type=int)
    parser.add_argument("--similarity_threshold", default=0.7, type=float)
    parser.add_argument("--mode", default=1, type=int)
    
    # for clustering
    parser.add_argument("--similarity_list_path", default="pkl/similarity.pickle", type=str)
    parser.add_argument("--limit_depth", default=5, type=int)
    parser.add_argument("--clustering_threshold", default=0.8, type=float)
    
    args = parser.parse_args()
    return args