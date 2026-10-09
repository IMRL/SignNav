import cv2
import ipdb
import pickle
import numpy as np
from gym import spaces
from fastdtw import fastdtw
from typing import Any, Dict, List
from scipy.spatial.distance import euclidean


def extract_instruction_tokens(observations: List[Dict], instruction_sensor_uuid: str, tokens_uuid: str = "tokens") -> Dict[str, Any]:
    """
    Extracts instruction tokens from an instruction sensor if the tokens exist and are in a dict structure.
    """
    
    if (instruction_sensor_uuid not in observations[0] or instruction_sensor_uuid == "pointgoal_with_gps_compass"):
        return observations
    
    for i in range(len(observations)):
        if (isinstance(observations[i][instruction_sensor_uuid], dict) and tokens_uuid in observations[i][instruction_sensor_uuid]):
            observations[i][instruction_sensor_uuid] = observations[i][instruction_sensor_uuid]["tokens"]
        else:
            break

    return observations


def extract_position_yaw(observations: List[Dict], global_gps_sensor_uuid: str):
    """
    Extract agent position and yaw from the global gps sensor if it exists.
    """
    
    if global_gps_sensor_uuid not in observations[0]:
        return observations
    
    for i in range(len(observations)):
        observations[i]["agent_position"] = observations[i][global_gps_sensor_uuid]["position"]
        observations[i]["agent_yaw"]      = observations[i][global_gps_sensor_uuid]["yaw"]
        del observations[i][global_gps_sensor_uuid]

    return observations


def single_frame_box_shape(box: spaces.Box) -> spaces.Box:
    """removes the frame stack dimension of a Box space shape if it exists."""
    if len(box.shape) < 4:
        return box

    return spaces.Box(low=box.low.min(), high=box.high.max(), shape=box.shape[1:], dtype=box.high.dtype)


# crop and resize the observations
def process_observations(observations):
    def crop_and_resize(img: np.ndarray, resize_height: int = 224):
        original_height, original_width = img.shape[:2] # 672, 896
        resize_ratio = original_height // resize_height # int: 3

        crop_height = (original_height // resize_ratio) * resize_ratio # int: 672 --> 672
        crop_width = (original_width // resize_ratio) * resize_ratio # int: 896 --> 894

        start_x = (original_width - crop_width) // 2 # int: 1
        start_y = (original_height - crop_height) // 2 # int: 0
        end_x = start_x + crop_width # 895
        end_y = start_y + crop_height # 672

        cropped_img = img[start_y:end_y, start_x:end_x] # img: [672, 896, 3], cropped_img: [672, 894, 3]
        resized_img = cv2.resize(cropped_img, (int(crop_width / resize_ratio), int(crop_height / resize_ratio))) # [672, 894, 3] --> [224, 298, 3]
        return resized_img
        
    for i, observation in enumerate(observations):
        hint_position = observation["hint_position"]
        top_left_corner = (hint_position[0], hint_position[1])
        bottom_right_corner = (hint_position[2], hint_position[3])
        
        rgb_image = observation["rgb"][..., 0:3][..., ::-1] # [672, 896, 3] change first to RGB and then change to BGR
        rgb_image = np.ascontiguousarray(rgb_image) # [672, 896, 3]
        cv2.rectangle(rgb_image, top_left_corner, bottom_right_corner, (0, 255, 0), 2) # [672, 896, 3]
        # rgb_image = crop_and_resize(rgb_image) # [224, 298, 3]
        rgb_image = cv2.resize(rgb_image, (224, 224)) # [224, 224, 3]
        
        depth_image = observation["depth"] # [672, 896, 1]
        # depth_image = crop_and_resize(depth_image) # [224, 298]
        depth_image = cv2.resize(depth_image, (224, 224)) # [224, 224]
        depth_image = depth_image.reshape(depth_image.shape[0], depth_image.shape[1], 1) # [224, 224, 1]

        hint_image = observation["hint"][..., 0:3][..., ::-1] # [224, 224, 3], change first to RGB and then change to BGR
        hint_image = np.ascontiguousarray(hint_image)

        observations[i]["rgb"]   = rgb_image
        observations[i]["depth"] = depth_image
        observations[i]["hint"]  = hint_image

    return observations


def save_agent_trajectory(agent_positions, agent_yaws, scene_id, episode_id, save_folder):
    positions = np.array(agent_positions) # [length, 2]
    yaws      = np.array(agent_yaws) # [length, ]

    traj_data = {"positions": positions,
                 "yaws": yaws,
                 }

    scene_id = scene_id.split("/")[-2]
    with open(f"{save_folder}/{scene_id}_{episode_id}.pkl", "wb") as f:
        pickle.dump(traj_data, f)


def calculate_ate(agent_trajectory, oracle_trajectory):
    """
    计算点数不一致时的绝对轨迹误差 (ATE)。
    采用“最近邻匹配”策略：对于机器人的每个点，寻找真值轨迹中几何距离最近的点进行匹配。

    Args:
        agent_trajectory (list of tuples/lists): 机器人估计轨迹 [(x, y), ...]
        oracle_trajectory (list of tuples/lists): 真值轨迹 [(x, y), ...]

    Returns:
        float: 均方根误差 (RMSE)
    """
    
    agent_arr = np.array(agent_trajectory)
    oracle_arr = np.array(oracle_trajectory)

    if len(agent_arr) == 0 or len(oracle_arr) == 0:
        raise ValueError("轨迹数据不能为空")

    # 3. 计算距离矩阵 (利用广播机制)
    # agent_arr 的形状扩充为 (M, 1, 2)
    # oracle_arr 的形状扩充为 (1, N, 2)
    # 差值 diff 的形状将为 (M, N, 2)，即计算了 M 个点到 N 个点的所有组合差值
    
    # 注意：如果轨迹非常长（例如超过1万个点），这种方式会消耗大量内存。
    # 此时建议使用 scipy.spatial.KDTree (代码末尾有说明)
    diff = agent_arr[:, np.newaxis, :] - oracle_arr[np.newaxis, :, :]
    
    # 4. 计算欧氏距离的平方
    # shape: (M, N)
    dist_squared_matrix = np.sum(diff**2, axis=2)

    # 5. 找到最近邻距离
    # 对于每一个 agent 点 (行)，找到最小的距离平方 (axis=1)
    min_dist_squared = np.min(dist_squared_matrix, axis=1)

    # 6. 计算 RMSE
    mse = np.mean(min_dist_squared)
    rmse = np.sqrt(mse)

    return rmse


def calculate_ate_dtw(agent_trajectory, oracle_trajectory):
    """
    使用动态时间规整 (DTW) 计算机器人轨迹与真值轨迹之间的绝对轨迹误差。
    
    该函数利用 fastdtw 算法在保持时间序列顺序的前提下，找到两条轨迹之间
    距离最小的非线性对齐路径，从而解决点数不一致和速度不均匀的问题。

    Args:
        agent_trajectory (list of tuples/lists): 机器人估计轨迹，格式为 [(x, y), ...]
        oracle_trajectory (list of tuples/lists): 真值轨迹，格式为 [(x, y), ...]

    Returns:
        dict: 包含以下统计信息的字典:
            - 'ate_rmse': 均方根误差 (Root Mean Square Error)
            - 'ate_mean': 平均误差 (Mean Error)
            - 'ate_std': 标准差 (Standard Deviation)
            - 'ate_max': 最大误差 (Max Error)
            - 'ate_min': 最小误差 (Min Error)
            - 'dtw_distance': DTW 算出的总累积距离 (Total Distance)
    """
    
    # 1. 数据预处理
    # 将列表转换为 NumPy 数组，确保数据格式正确
    agent_traj = np.array(agent_trajectory)
    oracle_traj = np.array(oracle_trajectory)

    # 数据有效性检查
    if len(agent_traj) < 1 or len(oracle_traj) < 1:
        raise ValueError("输入轨迹不能为空。")

    # 2. 运行 FastDTW 算法
    # dist: 是两条轨迹对齐后的总累积距离
    # path: 是一个列表，包含匹配的索引对 [(idx_agent_0, idx_oracle_0), (idx_agent_1, idx_oracle_1), ...]
    # dist=euclidean: 指定使用欧氏距离作为两个点之间的度量标准
    distance, path = fastdtw(agent_traj, oracle_traj, dist=euclidean)

    # 3. 提取逐点误差
    # DTW 返回的是总距离和路径，我们需要根据路径计算每一对匹配点的实际误差
    errors = []
    for i, j in path:
        # 获取匹配的点
        p_agent = agent_traj[i]
        p_oracle = oracle_traj[j]
        
        # 计算这两点间的欧氏距离
        # np.linalg.norm 计算向量的范数 (即距离)
        err = np.linalg.norm(p_agent - p_oracle)
        errors.append(err)

    errors = np.array(errors)

    # 4. 计算统计指标
    ate_rmse = np.sqrt(np.mean(errors**2))  # 均方根误差
    ate_mean = np.mean(errors)              # 平均误差
    ate_std = np.std(errors)                # 标准差
    ate_max = np.max(errors)                # 最大误差
    ate_min = np.min(errors)                # 最小误差

    return ate_rmse
    # return {
    #     'ate_rmse': ate_rmse,
    #     'ate_mean': ate_mean,
    #     'ate_std': ate_std,
    #     'ate_max': ate_max,
    #     'ate_min': ate_min,
    #     'dtw_distance': distance # 有时也需要参考总的DTW距离
    # }
        