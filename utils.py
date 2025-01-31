import os
from typing import List
import numpy as np
from scipy import signal
import scipy.io as sio
import pandas as pd
from event import Event
from arg import Arg


def mkdir(path: str) -> None:
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


def get_electrode_namelist(index: int, file_path='electrode_information.csv', event=True) -> List[str]:
    """
    获取指定数据集中，电极名称的顺序字符串，比如："FP1,FP3", 最后会添加一个event列(可选)
    :param event: 在列表的最后是否添加一个event列
    :param index: 数据集索引
    :param file_path: 电极信息文件路径
    :return: str
    """
    electrode_information = pd.read_csv(file_path)
    cols: pd.DataFrame = electrode_information.iloc[:, [0, 4 + index]]
    # 现在得到了第一列是电极名称，第二列是索引的列表，接下来删除0，并排序
    cols = cols[cols.iloc[:, 1] != 0]
    cols = cols.sort_values(by=cols.columns[1])
    ret: List[str] = cols.iloc[:, 0].tolist()
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


def filter_available_electrodes(df: pd.DataFrame, usage_threshold: int, has_event: bool, file_path='electrode_information.csv') -> pd.DataFrame:
    info = pd.read_csv(file_path)
    names: List[str] = info[info['Usage'] >= usage_threshold]['Electrode_name'].tolist()
    if has_event:
        names.append(Event.col_name.value)
    names = [name for name in names if name in df.columns]
    return df[names]


def preprocessing(df: pd.DataFrame, arg: Arg) -> pd.DataFrame:
    """
    降采样
    :param df: dataframe
    :param arg: 参数
    :return: 处理后的dataframe
    """
    n = df.shape[0]
    has_event = df.columns[-1] == Event.col_name.value

    if arg.use_available_electrodes:
        df = filter_available_electrodes(df, arg.usage_threshold, has_event)

    col = df.shape[1]

    assert arg.sampling_rate >= arg.resampling_rate
    if arg.resample and arg.sampling_rate > arg.resampling_rate:
        # 如果确认要降采样并且采样率前后不相等
        target_samples = int(n * arg.resampling_rate / arg.sampling_rate)
        if has_event:
            # 包含事件列，把事件与数据分离
            # 先对eeg降采样
            eeg = df.iloc[:, :col - 1]
            eeg = signal.resample(eeg, target_samples, axis=0)
            if arg.scale:
                # 如果缩放了，那么就趁现在缩放
                eeg = eeg * arg.scaler
            # 再对事件降采样
            event = df.iloc[:, col - 1].to_numpy()
            indices = np.linspace(0, n - 1, target_samples, dtype=int)
            event = event[indices]
            # 把eeg和事件合并
            eeg = np.hstack((eeg, event.reshape(-1, 1)))
        else:
            # 如果没有事件，那么就对全部eeg降采样
            eeg = signal.resample(df, target_samples, axis=0)
            if arg.scale:
                eeg = eeg * arg.scaler
        df = dataframe(eeg, df.columns, False)
    elif arg.scale:
        # 如果不做降采样，只缩放
        if has_event:
            df.iloc[:, :col - 1] = df.iloc[:, :col - 1] * arg.scaler
        else:
            df = df * arg.scaler
    return df
