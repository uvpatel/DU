import pandas as pd
import numpy as np

def load_csv(path):
    return pd.read_csv(path)

def basic_stats(df):
    return df.describe()

def handle_missing(df):
    return df.fillna(df.mean(numeric_only=True))