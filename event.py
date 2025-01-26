from enum import Enum


class Event(Enum):
    col_name = "Event"      # 事件列的列名称
    left_hand = 1           # 左手
    right_hand = 2          # 右手
    both_hand = 3           # 双手
    left_leg = 4            # 左腿
    right_leg = 5           # 右腿
    both_leg = 6            # 双腿
    tongue = 7              # 舌头
    feet = 8
    thumb = 9               # 大拇指
    index_finger = 10        # 食指
    middle_finger = 11      # 中指
    ring_finger = 12        # 无名指
    pinkie_finger = 13      # 小拇指
    passive = 14            # 虽然屏幕上出现提示，但是不进行想象
    # 对于session break、experiment end、initial relaxation，全部设置为0
