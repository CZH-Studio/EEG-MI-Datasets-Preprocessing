from dataclasses import dataclass


@dataclass
class Arg:
    # basic info
    index: int
    name: str
    folder_src: str
    folder_dst: str
    sampling_rate: int = 100  # 0 意味着数据集内可能有不同的采样率，需要手动设置

    # preprocessing
    preprocessing: bool = True              # 是否启用预处理
    use_available_electrodes: bool = True   # 使用大多数数据集都使用的电极
    usage_threshold: int = 4                # 使用数量阈值，8个数据集中，如果使用某一个电极的使用量小于这个阈值，那么会被删除
    resample: bool = True                   # 是否降采样
    resampling_rate: int = 100              # 目标采样率
    scale: bool = True                      # 是否对数据进行缩放（统一单位，建议统一为μV）
    scaler: float = 1.0                     # 缩放比例（乘scaler）
    result_sampling_rate: int = resampling_rate if preprocessing and resample else sampling_rate

    def set_sampling_rate(self, sampling_rate):
        self.sampling_rate = sampling_rate
        self.result_sampling_rate = self.resampling_rate if self.preprocessing and self.resample else self.sampling_rate
