# 数据集8：High-Gamma Dataset
import utils
from event import Event
import os
import mne
import numpy as np
from tqdm import tqdm
from arg import Arg


arg = Arg(
    index=8,
    name="High-Gamma Dataset",
    folder_src="E:/Code/Dataset/EEG/(8) High-Gamma Dataset/data",
    folder_dst="./(8) High-Gamma Dataset",
    sampling_rate=500,
    scaler=1e+6
)


def main():
    utils.mkdir(arg.folder_dst)

    file_list = (utils.list_dir(os.path.join(arg.folder_src, "train"), ".edf") +
                 utils.list_dir(os.path.join(arg.folder_src, "test"), ".edf"))
    total_person = 14
    for index, file in tqdm(enumerate(file_list), total=len(file_list), desc="正在处理数据集8"):
        # index用于判断是训练集还是测试集，训练集：0~13；测试集：14~27
        trial = 1 if 0 <= index <= total_person - 1 else 2    # trial：训练集1，测试集2
        person = index + 1 if 0 <= index <= total_person else index + 1 - total_person
        data = mne.io.read_raw_edf(file, preload=True, verbose=False)
        eeg: np.ndarray = data.get_data()
        eeg = np.delete(eeg, np.r_[32:37, 84:88, 92:96], axis=0)     # [120, n]

        # 处理event
        events_raw = data.annotations.description
        events_onset = data.annotations.onset
        # 这里不需要获取duration，因为所有的duration都是4s
        events = np.zeros(eeg.shape[1])
        assert events_raw.shape[0] == events_onset.shape[0]
        for i in range(events_raw.shape[0]):
            event_name = events_raw[i]
            event_onset = int(events_onset[i] * arg.sampling_rate)
            match event_name:
                case 'rest':
                    events[event_onset: event_onset + 4 * arg.sampling_rate] = Event.passive.value
                case 'feet':
                    events[event_onset: event_onset + 4 * arg.sampling_rate] = Event.feet.value
                case 'left_hand':
                    events[event_onset: event_onset + 4 * arg.sampling_rate] = Event.left_hand.value
                case 'right_hand':
                    events[event_onset: event_onset + 4 * arg.sampling_rate] = Event.right_hand.value
                case _:
                    pass

        # 保存
        eeg = np.vstack((eeg, events.reshape(1, -1))).T
        electrode_namelist = utils.get_electrode_namelist(8)
        df = utils.dataframe(eeg, electrode_namelist)
        if arg.preprocessing:
            df = utils.preprocessing(df, arg)
        utils.save_parquet(df, os.path.join(arg.folder_dst, f"{arg.index:02d}_{person:02d}_{trial:02d}_{arg.result_sampling_rate}.parquet"))


if __name__ == '__main__':
    main()
