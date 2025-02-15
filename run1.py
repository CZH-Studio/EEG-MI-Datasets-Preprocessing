# 数据集1：Left_Right Hand MI，单位应该是uV，应该转换成mV计算比较好
import os
from tqdm import tqdm
import utils
from event import Event
import numpy as np
from arg import Arg


arg = Arg(
    index=1,
    name="Left_Right Hand MI",
    folder_src="E:/Code/Dataset/EEG/(1) Left_Right Hand MI",
    folder_dst="./(1) Left_Right Hand MI",
    sampling_rate=512,
    scaler=1e-4
)


def main():
    utils.mkdir(arg.folder_dst)

    # 获取文件夹下所有.mat文件
    file_list = utils.list_dir(arg.folder_src, '.mat')
    # 获取电极名称
    electrode_namelist = utils.get_electrode_namelist(1)
    for person, file in tqdm(enumerate(file_list), total=len(file_list), desc=f"正在处理数据集{arg.index}"):
        data = utils.load_mat(file)['eeg']
        # 对于左右手的想象，保存为不同的文件
        imagery_left: np.ndarray = data['imagery_left'][0][0]
        imagery_left = imagery_left[:64, :]   # 去除EMG通道
        imagery_right: np.ndarray = data['imagery_right'][0][0]
        imagery_right = imagery_right[:64, :]
        imagery_event: np.ndarray = data['imagery_event'][0][0][0].astype(np.uint8)

        # 原始的事件数组只是想象运动的开始，现在希望把想象过程中的所有的标签值都设置为1，想象运动持续3秒，采样率为512
        # 那么就需要让每一个1后面跟随3*512=1536个1（左手想象）或2（右手想象）
        indices = np.where(imagery_event == 1)[0]
        imagery_event_left = np.zeros(imagery_event.shape)
        imagery_event_right = np.zeros(imagery_event.shape)
        for i in indices:
            imagery_event_left[i:i+3 * arg.sampling_rate] = Event.left_hand.value
            imagery_event_right[i:i+3 * arg.sampling_rate] = Event.right_hand.value

        # 将处理好的事件数组添加到eeg上，别忘了把标签转换为整形
        imagery_left = np.vstack((imagery_left, imagery_event_left)).T
        imagery_right = np.vstack((imagery_right, imagery_event_right)).T
        df_left = utils.dataframe(imagery_left, electrode_namelist)
        df_right = utils.dataframe(imagery_right, electrode_namelist)

        # 开启预处理
        if arg.preprocessing:
            df_left = utils.preprocessing(df_left, arg)
            df_right = utils.preprocessing(df_right, arg)

        utils.save_parquet(df_left, os.path.join(arg.folder_dst, f'{arg.index:02d}_{person+1:02d}_01_{arg.result_sampling_rate}.parquet'))
        utils.save_parquet(df_right, os.path.join(arg.folder_dst, f'{arg.index:02d}_{person+1:02d}_02_{arg.result_sampling_rate}.parquet'))


if __name__ == '__main__':
    main()
