# 数据集2：Motor Movement_Imagery Dataset
import numpy as np
from tqdm import tqdm
import utils
from event import Event
import os
import mne
from arg import Arg


arg = Arg(
    index=2,
    name="Motor Movement_Imagery Dataset",
    folder_src="E:/Code/Dataset/EEG/(2) Motor Movement_Imagery Dataset",
    folder_dst="./(2) Motor Movement_Imagery Dataset",
    sampling_rate=160,
    scaler=1e+6
)


def main():
    utils.mkdir(arg.folder_dst)

    # 遍历每一个受试者（文件夹），每个文件夹下有14个EDF文件，其中只有一些是有关运动想象的，其他不要了
    electrode_namelist = utils.get_electrode_namelist(2)
    enabled_index = [3, 5, 7, 9, 11, 13]
    sampling_rate = 160
    folder_list = utils.list_dir(arg.folder_src, folder_only=True)
    progressbar = tqdm(total=len(folder_list) * len(enabled_index), desc=f"正在处理数据集{arg.index}")
    for person, folder in enumerate(folder_list):
        file_list = utils.list_dir(folder, ".edf")
        file_list = [file_list[i] for i in enabled_index]
        for trial, file in enumerate(file_list):
            # 从EDF文件中获取EEG数据
            edf = mne.io.read_raw_edf(file, preload=True, verbose=False)
            eeg = edf.get_data()
            # 对于事件，需要从annotations中获取
            # 其中，对于trial = [3, 4, 7, 8, 11, 12]，T1代表左拳(1)，T2代表右拳(2)
            # 对于trial = [5, 6, 9, 10, 13, 14]，T1代表双拳(3)，T2代表双脚(4)
            onset_list = edf.annotations.onset
            duration_list = edf.annotations.duration
            description_list = edf.annotations.description
            event = np.zeros(eeg.shape[1], dtype=np.uint8)
            for i in range(onset_list.shape[0]):
                onset = onset_list[i]
                duration = duration_list[i]
                match str(description_list[i]):
                    case 'T0':
                        pass
                    case 'T1':
                        if trial in [0, 2, 4]:
                            event[int(onset * sampling_rate): int((onset + duration) * sampling_rate)] = Event.left_hand.value
                        else:
                            event[int(onset * sampling_rate): int((onset + duration) * sampling_rate)] = Event.both_hand.value
                    case 'T2':
                        if trial in [0, 2, 4]:
                            event[int(onset * sampling_rate): int((onset + duration) * sampling_rate)] = Event.right_hand.value
                        else:
                            event[int(onset * sampling_rate): int((onset + duration) * sampling_rate)] = Event.both_leg.value

            # 将标签添加到eeg的最后一行，然后整体转置
            eeg = np.vstack((eeg, event)).T
            df = utils.dataframe(eeg, electrode_namelist)

            if arg.preprocessing:
                df = utils.preprocessing(df, arg)
            utils.save_parquet(df,
                               os.path.join(arg.folder_dst, f"{arg.index:02d}_{person+1:02d}_{trial+1:02d}_{arg.result_sampling_rate}.parquet"))
            progressbar.update(1)


if __name__ == '__main__':
    main()
