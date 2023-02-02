import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--key_path", default="keys/rayn_key.json", type=str)
    parser.add_argument("--mode", default=1, type=int)
    parser.add_argument("--blob_name", default="saved_model/", type=str)
    parser.add_argument("--file_path", default="../model/saved_model/best_model.pt", type=str)
    parser.add_argument("--bucket_name", default="model_storage_rayn", type=str)
    parser.add_argument("--save_path", default="../backend/inference/best_model.pt", type=str)
    parser.add_argument("--bucket_location", default="US", type=str)
    args = parser.parse_args()
    return args