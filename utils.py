import os
from typing import List
import numpy as np
import scipy.io as sio
import pandas as pd
from event import Event


def mkdir(path) -> None:
    """
    Create a folder if not exists
    :param path:
    :return:
    """
    if not os.path.exists(path):
        os.makedirs(path)


def list_dir(folder_path: str, ext=None, folder_only=False, file_only=False) -> List[str]:
    """
    Get a list of all files in a folder
    :param file_only: 只获取文件名
    :param folder_only: 只获取文件夹名
    :param folder_path: 文件夹路径
    :param ext: 扩展名
    :return: 文件列表
    """
    file_list = os.listdir(folder_path)
    if ext:
        if isinstance(ext, str):
            ext = [ext]
        file_list = [f for f in file_list if os.path.splitext(f)[1] in ext]
    # 将每一个文件名前添加路径
    file_list = [os.path.join(folder_path, f) for f in file_list]
    # 过滤文件夹
    if folder_only:
        file_list = [f for f in file_list if os.path.isdir(f)]
    # 过滤文件
    if file_only:
        file_list = [f for f in file_list if not os.path.isdir(f)]
    return file_list


def load_mat(file_path: str) -> dict:
    """
    Load mat file
    :param file_path: 文件路径
    :return: 数据
    """
    return sio.loadmat(file_path)


def get_electrode_namelist(index: int, file_path='electrode_information.csv', event=True) -> list:
    """
    获取指定数据集中，电极名称的顺序字符串，比如："FP1,FP3", 最后会添加一个event列(可选)
    :param event: 在列表的最后是否添加一个event列
    :param index: 数据集索引
    :param file_path: 电极信息文件路径
    :return: str
    """
    electrode_information = pd.read_csv(file_path)
    cols: pd.DataFrame = electrode_information.iloc[:, [0, 4+index]]
    # 现在得到了第一列是电极名称，第二列是索引的列表，接下来删除NA，并排序
    cols = cols.dropna(subset=[cols.columns[1]])
    cols = cols.sort_values(by=cols.columns[1])
    ret: list = cols.iloc[:, 0].tolist()
    if event:
        ret.append(Event.col_name.value)
    return ret


def dataframe(data: np.ndarray, cols: list, event_col_to_int=True) -> pd.DataFrame:
    """
    创建一个dataframe
    :param event_col_to_int: 通常最后一列作为标签，那么是否将最后一列的数据类型转换为uint8（只有最后一列名称是Event时，才会生效）
    :param data: ndarray
    :param cols: 列名称
    :return: DataFrame
    """
    df = pd.DataFrame(data, columns=cols)
    event_col_name = Event.col_name.value
    if event_col_to_int:
        if event_col_name in df.columns:
            df[event_col_name] = df[event_col_name].astype(np.uint8)
    return df


def save_csv(df: pd.DataFrame, file_path: str, index=False) -> None:
    """
    保存一个dataframe为csv
    :param df: dataframe
    :param file_path: 路径
    :param index: 是否保存索引，默认False
    :return: None
    """
    df.to_csv(file_path, index=index)


def save_parquet(df: pd.DataFrame, file_path: str, index=False) -> None:
    """
    保存一个dataframe为pickle
    :param df: dataframe
    :param file_path: 路径
    :param index: 是否保存索引，默认False
    :return: None
    """
    df.to_parquet(file_path, index=index)


