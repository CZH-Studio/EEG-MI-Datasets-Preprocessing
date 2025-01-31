# 数据集4：The largest SCP data of Motor-Imagery，单位：mV，数值基本上是0~20左右
import utils
from event import Event
import os
import numpy as np
from tqdm import tqdm
from arg import Arg


arg = Arg(
    index=4,
    name="The largest SCP data of Motor-Imagery",
    folder_src="E:/Code/Dataset/EEG/(4) The largest SCP data of Motor-Imagery",
    folder_dst="./(4) The largest SCP data of Motor-Imagery",
    sampling_rate=0,
    scale=False
)


def main():
    utils.mkdir(arg.folder_dst)

    # 获取文件夹下所有.mat文件
    file_list = utils.list_dir(arg.folder_src, '.mat')
    # 获取电极名称
    electrode_namelist = utils.get_electrode_namelist(4)
    # 实验名称与序号之间的映射
    trial_mapping = {
        'CLA': '01',
        'HaLT': '02',
        '5F': '03',
        'FREEFORM': '04',
        'NoMT': '05'
    }

    for file in tqdm(file_list, total=len(file_list), desc=f"正在处理数据集{arg.index}"):
        data = utils.load_mat(file)['o']
        file_name = os.path.basename(file)
        file_info = file_name.split('-')
        trial_type = file_info[0]

        # 判断人是否在计数器中，如果不在就添加进来
        person = ord(file_info[1][-1].upper()) - ord('A') + 1

        # 获取不同通道的数据，其中只有前21个通道是eeg数据
        marker = data['marker'][0][0]
        eeg = data['data'][0][0][:, :21]
        arg.set_sampling_rate(data['sampFreq'][0][0][0][0])

        # 处理marker
        if trial_type == '5F':
            mapping = {0: 0,
                       1: Event.thumb.value,
                       2: Event.index_finger.value,
                       3: Event.middle_finger.value,
                       4: Event.ring_finger.value,
                       5: Event.pinkie_finger.value,
                       90: 0,
                       91: 0,
                       92: 0,
                       99: 0}
        else:
            mapping = {0: 0,
                       1: Event.left_hand.value,
                       2: Event.right_hand.value,
                       3: Event.passive.value,
                       4: Event.left_leg.value,
                       5: Event.tongue.value,
                       6: Event.right_leg.value,
                       90: 0,
                       91: 0,
                       92: 0,
                       99: 0}
        vectorized_mapping = np.vectorize(mapping.get)
        marker = vectorized_mapping(marker)

        # 将marker添加到eeg的最后一列
        eeg = np.hstack((eeg, marker))
        # 保存数据
        assert eeg.shape[1] == len(electrode_namelist)
        df = utils.dataframe(eeg, electrode_namelist)
        if arg.preprocessing:
            df = utils.preprocessing(df, arg)
        utils.save_parquet(df, os.path.join(arg.folder_dst, f"{arg.index:02d}_{person:02d}_{trial_mapping[trial_type]}_{arg.result_sampling_rate}.parquet"))


if __name__ == '__main__':
    main()
