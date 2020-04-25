import pandas as pd
import numpy as np
import os


"""
This module is used to do log record. Backend library is pandas.

"""


def record_csv(epoch, log_columns, save_path, filename):
    path = os.path.join(save_path, filename)
    if isinstance(log_columns, list) and epoch == 1:
        df = pd.DataFrame(data=[], columns=log_columns)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        df.to_csv(path, index=False)
    elif isinstance(log_columns, dict):
        df = pd.DataFrame(log_columns, index=[epoch-1])
        df.to_csv(path, mode='a', header=False, index=False)