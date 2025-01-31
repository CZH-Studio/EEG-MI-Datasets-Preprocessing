# BCI Competition IV-1，单位：1e-7 V（即0.1uV）
# 请注意！以05_xx_02_xxx.parquet命名的文件，不包含event，只有数据
import numpy as np
import os
import utils
from event import Event
from tqdm import tqdm
from arg import Arg


arg = Arg(
    index=5,
    name="BCI Competition IV-1",
    folder_src="E:/Code/Dataset/EEG/(5) BCI Competition IV-1",
    folder_dst="./(5) BCI Competition IV-1",
    sampling_rate=0,
    scale=False
)


def main():
    folder_src = "E:/Code/Dataset/EEG/(5) BCI Competition IV-1"
    folder_dst = r"./(5) BCI Competition IV-1"
    utils.mkdir(folder_dst)

    file_list = utils.list_dir(folder_src, ".mat")

    # 遍历文件
    for file in tqdm(file_list, total=len(file_list), desc=f"正在处理数据集{arg.index}"):
        mat = utils.load_mat(file)
        eeg = mat['cnt']                        # n*59

        info = mat['nfo']
        classes = info[0][0]['classes'][0]
        arg.set_sampling_rate(info[0][0]['fs'][0][0])   # 采样率
        file_name = os.path.basename(file)
        file_info = file_name.split('_')
        person = ord(file_info[2][3]) - ord('a') + 1

        # 先处理eeg数据，删除不需要的电极
        # 由于第17（CFC7）、24（CFC8）、34（CCP7）、41（CCP8）并没有找到名称以及位置信息，不在10-05标准内，因此放弃这些电极
        electrode_to_delete = [16, 23, 33, 40]
        eeg = np.delete(eeg, electrode_to_delete, 1)

        if file_info[1] == 'calib':
            # 如果是训练数据，那么其中包含标签
            trial = 1
            electrode_namelist = utils.get_electrode_namelist(5, event=True)

            # 处理mark
            mark_raw = mat['mrk']  # 原始标记数组
            mark_pos = mark_raw[0][0]['pos'][0]
            mark_label = mark_raw[0][0]['y'][0]
            mark = np.zeros(eeg.shape[0])  # 新的标记数组
            mark_mapping = {-1: 0, 1: 1}  # mark_label的-1代表第一类，1代表第二类
            for i in range(mark_pos.shape[0]):
                class_name = classes[mark_mapping[mark_label[i]]][0]  # left / right / foot
                event_start = mark_pos[i]
                match class_name:
                    case 'left':
                        mark[event_start: event_start + 400] = Event.left_hand.value
                    case 'right':
                        mark[event_start: event_start + 400] = Event.right_hand.value
                    case 'foot':
                        mark[event_start: event_start + 400] = Event.feet.value
            eeg = np.hstack((eeg, mark.reshape(-1, 1)))
            df = utils.dataframe(eeg, electrode_namelist)

        else:
            # 如果是验证数据，那么不包含标签
            trial = 2
            electrode_namelist = utils.get_electrode_namelist(5, event=False)
            df = utils.dataframe(eeg, electrode_namelist, event_col_to_int=False)

        if arg.preprocessing:
            df = utils.preprocessing(df, arg)
        utils.save_parquet(df, os.path.join(folder_dst, f"{arg.index:02d}_{person:02d}_{trial:02d}_{arg.result_sampling_rate}.parquet"))


if __name__ == '__main__':
    main()
