# 数据集6：BCI Competition IV-2a，单位：100uV
import utils
from event import Event
from tqdm import tqdm
import numpy as np
import os
from arg import Arg


arg = Arg(
    index=6,
    name="BCI Competition IV-2a",
    folder_src="E:/Code/Dataset/EEG/(6) BCI Competition IV-2a/npz",
    folder_dst="./(6) BCI Competition IV-2a",
    sampling_rate=250,
    scale=False
)


def main():
    utils.mkdir(arg.folder_dst)
    file_list = utils.list_dir(arg.folder_src, '.npz')

    for file in tqdm(file_list, total=len(file_list), desc=f'正在处理数据集{arg.index}'):
        # 读取数据
        data = np.load(file)
        eeg = data['s']

        # 读取数据信息
        file_name = os.path.basename(file)
        person = int(file_name[2])
        trial = 1 if file_name[3] == 'T' else 2     # 1:train 2:test

        # 删除3个EOG通道
        eeg = eeg[:, :22]

        # 如果数据是训练集，那么处理事件
        if trial == 1:
            event_type = data['etyp'].ravel()
            event_onset = data['epos'].ravel()
            event = np.zeros(eeg.shape[0])

            for i in range(event_type.shape[0]):
                onset = event_onset[i] - 1  # -1是因为原始数组的位置是从1开始的
                type_ = event_type[i]
                # onset是屏幕上出现提示的时间，实际开始想象的时间是onset+1~onset+4
                match type_:
                    case 769:
                        event[onset + arg.sampling_rate: onset + arg.sampling_rate * 4] = Event.left_hand.value
                    case 770:
                        event[onset + arg.sampling_rate: onset + arg.sampling_rate * 4] = Event.right_hand.value
                    case 771:
                        event[onset + arg.sampling_rate: onset + arg.sampling_rate * 4] = Event.feet.value
                    case 772:
                        event[onset + arg.sampling_rate: onset + arg.sampling_rate * 4] = Event.tongue.value
                    case _:
                        pass

            # 然后把列添加到eeg上
            eeg = np.hstack((eeg, event.reshape(-1, 1)))
            electrode_namelist = utils.get_electrode_namelist(6, event=True)
            assert eeg.shape[1] == len(electrode_namelist)
        else:
            # 如果是测试集，则不需要考虑event列
            electrode_namelist = utils.get_electrode_namelist(6, event=False)
            assert eeg.shape[1] == len(electrode_namelist)

        # 保存
        df = utils.dataframe(eeg, electrode_namelist)
        if arg.preprocessing:
            df = utils.preprocessing(df, arg)
        utils.save_parquet(df, os.path.join(arg.folder_dst, f"{arg.index:02d}_{person:02d}_{trial:02d}_{arg.result_sampling_rate}.parquet"))


if __name__ == '__main__':
    main()
