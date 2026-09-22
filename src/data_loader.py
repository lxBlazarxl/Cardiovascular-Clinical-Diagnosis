import urllib.request

import numpy as np
import pandas as pd

from src import config


def download_raw_data(force: bool = False) -> None:
    config.DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    if config.RAW_DATA_PATH.exists() and not force:
        return
    with urllib.request.urlopen(config.RAW_DATA_URL, timeout=60) as response:
        payload = response.read()
    config.RAW_DATA_PATH.write_bytes(payload)


def load_raw_data(force_download: bool = False) -> pd.DataFrame:
    download_raw_data(force=force_download)
    df = pd.read_csv(
        config.RAW_DATA_PATH,
        header=None,
        names=config.COLUMN_NAMES,
        na_values=config.MISSING_TOKEN,
    )
    for column in config.NUMERIC_COLUMNS + config.BINARY_COLUMNS + [config.TARGET_COLUMN]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    for column in config.CATEGORICAL_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df[config.TARGET_COLUMN] = (df[config.TARGET_COLUMN] > 0).astype(int)
    return df


def split_features_target(df: pd.DataFrame):
    X = df.drop(columns=[config.TARGET_COLUMN])
    y = df[config.TARGET_COLUMN].astype(int)
    return X, y
