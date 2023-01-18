import os
import time
import pandas as pd

def logging_time(original_fn):
    def wrapper_fn(*args, **kwargs):
        start_time = time.time()
        result = original_fn(*args, **kwargs)
        end_time = time.time()
        print("WorkingTime[{}]: {} sec".format(original_fn.__name__, end_time-start_time))
        return result
    return wrapper_fn

@logging_time
def merge_all_in_dir(dir):
    file_list = os.listdir(dir)
    files = list(map(lambda x: pd.read_csv(os.path.join(dir, x)),file_list))
    return pd.concat(files).drop_duplicates()
