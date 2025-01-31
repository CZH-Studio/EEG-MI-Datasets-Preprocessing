# 数据集7：BCI Competition IV-2b，单位：100uV
import utils
from event import Event
from tqdm import tqdm
import numpy as np
import os
import mne
from arg import Arg


arg = Arg(
    index=7,
    name="BCI Competition IV-2b",
    folder_src="E:/Code/Dataset/EEG/(7) BCI Competition IV-2b",
    folder_dst="./(7) BCI Competition IV-2b",
    sampling_rate=250,
    scaler=1e+6
)


def main():
    utils.mkdir(arg.folder_dst)
    file_list = utils.list_dir(arg.folder_src, '.gdf')

    for file in tqdm(file_list, total=len(file_list), desc=f'正在处理数据集{arg.index}'):
        # 获取文件信息
        file_name = os.path.basename(file)
        person = int(file_name[1:3])
        trial = int(file_name[3:5])      # 1,2,3:train  4,5:test

        # 读取gdf数据
        # 注意，此行代码要在numpy版本为1.x下执行，不能numpy=2
        gdf = mne.io.read_raw_gdf(file, preload=True, verbose=False)
        eeg = gdf.get_data()
        eeg = eeg[:3, :]    # [3, n]

        if 1 <= trial <= 3:
            # 训练集，含event
            events_raw = gdf.annotations.description
            events_duration = gdf.annotations.duration
            events_onset = gdf.annotations.onset
            events = np.zeros(eeg.shape[1])
            assert len(events_raw) == len(events_duration) == len(events_onset)
            for i in range(len(events_raw)):
                event_id = events_raw[i]
                onset = int(events_onset[i] * arg.sampling_rate)
                match event_id:
                    case 769:
                        if 1 <= trial <= 2:
                            events[onset: onset + int(3 * arg.sampling_rate)] = Event.left_hand.value
                        else:
                            events[onset: onset + int(3.5 * arg.sampling_rate)] = Event.left_hand.value
                    case 770:
                        if 1 <= trial <= 2:
                            events[onset: onset + int(3 * arg.sampling_rate)] = Event.right_hand.value
                        else:
                            events[onset: onset + int(3.5 * arg.sampling_rate)] = Event.right_hand.value
                    case _:
                        pass
            eeg = np.vstack((eeg, events.reshape(1, -1))).T
            electrode_namelist = utils.get_electrode_namelist(7, event=True)
            assert eeg.shape[1] == len(electrode_namelist)
        else:
            # 测试集，没有events
            eeg = eeg.T
            electrode_namelist = utils.get_electrode_namelist(7, event=False)
            assert eeg.shape[1] == len(electrode_namelist)

        # 创建dataframe，保存文件
        df = utils.dataframe(eeg, electrode_namelist)
        if arg.preprocessing:
            df = utils.preprocessing(df, arg)
        utils.save_parquet(df, os.path.join(arg.folder_dst, f"{arg.index:02d}_{person:02d}_{trial:02d}_{arg.result_sampling_rate}.parquet"))


if __name__ == '__main__':
    main()
