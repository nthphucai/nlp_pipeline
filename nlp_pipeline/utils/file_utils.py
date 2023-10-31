import datetime
import glob
import json
import logging
import os
import pathlib
import pickle
import shutil
from typing import List, Optional

import pandas as pd
import yaml

from constant import PRETRAINED_MODEL_URL_MAP


# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.INFO)


# Access to Yaml File
def read_yaml_file(file_path: str):
    with open(file_path, "r") as f:
        config = yaml.load(f, yaml.FullLoader)
    return config


def read_df_file(file_path: str):
    return pd.read_csv(file_path)


# access to text files
def read_text_file(file_path: str) -> List[str]:
    with open(file_path, "r") as f:
        context = [item.strip() for item in f.readlines()]
    return context


def write_text_file(context: List[str], file_path: str):
    with open(file_path, "w") as f:
        for text in context:
            f.write(text + "\n")


# remove all files in path
def remove_files(path):
    files = glob.glob(f"{path}/*")
    for f in files:
        os.remove(f)


# access to Picke File
def write_pickle_file(data: dict, path: str, name: Optional[str] = None) -> None:
    if name is not None:
        save_path = os.path.join(f"{path}", f"{name}" + ".pkl")
    else:
        save_path = path

    f = open(save_path, "wb")
    pickle.dump(data, f)
    f.close()


def read_pickle_file(path, name: Optional[str] = None) -> dict:
    if name is not None:
        save_path = os.path.join(f"{path}", f"{name}" + ".pkl")
    else:
        save_path = path

    f = open(save_path, "rb")
    pickle_file = pickle.load(f)
    return pickle_file


# access to Json File
def write_json_file(
    data: dict, path: str, name: Optional[str] = None, **kwargs
) -> None:
    if name is not None:
        save_path = os.path.join(f"{path}", f"{name}" + ".json")
    else:
        save_path = path
    with open(save_path, "w") as outfile:
        json.dump(data, outfile, ensure_ascii=False, indent=4, **kwargs)


def load_json_file(path: str, name: Optional[str] = None, **kwargs) -> dict:
    if name is not None:
        save_path = os.path.join(f"{path}", f"{name}" + ".json")
    else:
        save_path = path
    with open(save_path, encoding="utf-8") as outfile:
        data = json.load(outfile, **kwargs)
        return data


def load_dataframe_file(path: str, convert_to_json: bool = True):
    file_extension = pathlib.Path(path).suffix
    if ".xlsx" in file_extension:
        data = pd.read_excel(path, sheet_name=["Template"])
    elif ".csv" in file_extension:
        data = pd.read_csv(path)
    else:
        raise NotImplementedError("only support .csv and .xlsx extension")

    if convert_to_json:
        data = pd.DataFrame(data["context"])
        data.dropna(inplace=True, how="all")
        data = pd.DataFrame(pd.unique(data["context"]), data=["context"])
        data = data.reset_index(drop=True)
        data = data.to_dict("records")
    return data


def download_from_minio_url(url, save_path):
    os.system(f"wget --progress=bar:force:noscroll {url} -P {save_path}")
    zip_filename = os.path.basename(os.path.normpath(url))
    return zip_filename


def unzip(zip_path, extract_path):
    shutil.unpack_archive(filename=zip_path, extract_dir=extract_path)


def download_trained_model(domain: str, task: str, save_path: str):
    """
    Download multitask model or multichoice model of QuesGen
    """
    model_url = PRETRAINED_MODEL_URL_MAP[domain][task]

    downloaded_file = download_from_minio_url(url=model_url, save_path=save_path)
    unzip(zip_path=os.path.join(save_path, downloaded_file), extract_path=save_path)
    os.system(f"rm -f {os.path.join(save_path, downloaded_file)}")
    logger.info(f"Multitask model is downloaded! Save at {save_path}")


def get_time() -> str:
    """
    Return the current time.
    Returns:
        str: The string of current time.
    """
    return datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")


def format_arg_str(args, max_len: int = 50) -> str:
    """
    Beauty arguments.
    Args:
        args: Input arguments.
        max_len (int): Max length of printing string.

    Returns:
        str: Output beautied arguments.
    """
    linesep = os.linesep
    arg_dict = vars(args)
    keys = [k for k in arg_dict.keys()]
    values = [arg_dict[k] for k in keys]
    key_title, value_title = "Arguments", "Values"
    key_max_len = max(map(lambda x: len(str(x)), keys))
    value_max_len = min(max(map(lambda x: len(str(x)), values)), max_len)
    key_max_len, value_max_len = (
        max([len(key_title), key_max_len]),
        max([len(value_title), value_max_len]),
    )
    horizon_len = key_max_len + value_max_len + 5
    res_str = linesep + "=" * horizon_len + linesep
    res_str += (
        " "
        + key_title
        + " " * (key_max_len - len(key_title))
        + " | "
        + value_title
        + " " * (value_max_len - len(value_title))
        + " "
        + linesep
        + "=" * horizon_len
        + linesep
    )
    for key in sorted(keys):
        value = arg_dict[key]
        if value is not None:
            key, value = str(key), str(value).replace("\t", "\\t")
            value = value[: max_len - 3] + "..." if len(value) > max_len else value
            res_str += (
                " "
                + key
                + " " * (key_max_len - len(key))
                + " | "
                + value
                + " " * (value_max_len - len(value))
                + linesep
            )
    res_str += "=" * horizon_len
    return res_str
