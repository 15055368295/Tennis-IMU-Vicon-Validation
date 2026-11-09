# -*- coding: utf-8 -*-

import os
import re
import io
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
# ============================================================
# 1. 用户配置：路径和动作类型
# ============================================================
VICON_DIR = r"E:\Vicon数据\Viocn-未截取"
IMU_DIR = r"E:\IMU数据\IMU-未截取"
VICON_TIME_FILE = r"E:\Vicon数据\Viocn-未截取\split_cycle_time-GZ.xlsx"
IMU_TIME_FILE = r"E:\IMU数据\IMU-未截取\cycle time GZ.xlsx"
SAVE_DIR = r"E:\RMSE分析结果\GZ_Aligned_1000Points"

# 根据当前时间表所对应的动作类型设置：
# 正手 forehand -> 骨盆旋转最小值
# 反手 backhand -> 骨盆旋转最大值
STROKE_TYPE = "backhand"  

TARGET_POINTS = 1000
EXPECTED_FS = 100.0
TIME_TOLERANCE = 0.03 

os.makedirs(SAVE_DIR, exist_ok=True)

# 11个DOF映射
JOINT_MAPPING = {
    "Pelvis Tilt": {
        "vicon": "pelvis_tilt",
        "imu": "Pelvis Tilt",
    },
    "Pelvis Obliquity": {
        "vicon": "pelvis_list",
        "imu": "Pelvis Obliquity",
    },
    "Pelvis Rotation": {
        "vicon": "pelvis_rotation",
        "imu": "Pelvis Rotation",
    },
    "R-Hip Flex/Ext": {
        "vicon": "hip_flexion_r",
        "imu": "Right Hip Flexion-Extension",
    },
    "R-Hip Addu/Abdu": {
        "vicon": "hip_adduction_r",
        "imu": "Right Hip Adduction-Abduction",
    },
    "R-Hip Rotation": {
        "vicon": "hip_rotation_r",
        "imu": "Right Hip Rotation",
    },
    "R-Knee Flex/Ext": {
        "vicon": "knee_angle_r",
        "imu": "Right Knee Flexion-Extension",
    },
    "L-Hip Flex/Ext": {
        "vicon": "hip_flexion_l",
        "imu": "Left Hip Flexion-Extension",
    },
    "L-Hip Addu/Abdu": {
        "vicon": "hip_adduction_l",
        "imu": "Left Hip Adduction-Abduction",
    },
    "L-Hip Rotation": {
        "vicon": "hip_rotation_l",
        "imu": "Left Hip Rotation",
    },
    "L-Knee Flex/Ext": {
        "vicon": "knee_angle_l",
        "imu": "Left Knee Flexion-Extension",
    },
}

JOINT_ORDER = list(JOINT_MAPPING.keys())

# 与原始代码一致的IMU坐标方向修正
IMU_SIGN_FLIP_JOINTS = ["Pelvis Tilt"]


# ============================================================
# 2. 基础工具函数
# ============================================================

def normalize_name(value):
    """列名/文件名标准化，便于模糊匹配。"""
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def standardize_id(name):
    """提取文件名中的编号，尽量匹配Vicon和IMU试次。"""
    text = str(name).lower()
    alpha_num = re.findall(r"[a-z]+\d+", text)
    digits = re.findall(r"\d+", text)
    return f"{alpha_num[0] if alpha_num else ''}_{digits[-1] if digits else ''}"


def smart_get_col(df, keyword):
    target = normalize_name(keyword)
    for col in df.columns:
        if target in normalize_name(col):
            return col
    return None


def find_file_fuzzy(root, action_id, extension):
    """在根目录及其子目录中查找匹配文件。"""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"数据目录不存在：{root}")

    target = normalize_name(action_id)
    candidates = list(root.rglob(f"*{extension}"))

    # 优先按完整标准化文件名匹配
    for path in candidates:
        if target and target in normalize_name(path.stem):
            return str(path)

    # 再按试次编号匹配
    target_id = standardize_id(action_id)
    for path in candidates:
        if standardize_id(path.stem) == target_id:
            return str(path)

    return None


def read_vicon_mot(filepath):
    """读取OpenSim/Vicon导出的MOT文件，兼容空格或Tab分隔。"""
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    header_line = None
    for idx, line in enumerate(lines):
        if "endheader" in line.lower():
            header_line = idx + 1
            break

    if header_line is None:
        raise ValueError(f"MOT文件中未找到 endheader：{filepath}")

    df = pd.read_csv(
        io.StringIO("".join(lines[header_line:])),
        sep=r"\s+",
        engine="python",
    )
    df.columns = [str(c).strip() for c in df.columns]
    return df


def clean_sort_time_series(df, time_col):
    """删除无效时间戳、按时间排序，并去除重复时间戳。"""
    if time_col not in df.columns:
        raise KeyError(f"缺少时间列：{time_col}")

    out = df.copy()
    out[time_col] = pd.to_numeric(out[time_col], errors="coerce")
    out = out.loc[np.isfinite(out[time_col].to_numpy())].copy()
    out = out.sort_values(time_col).drop_duplicates(subset=time_col, keep="first")
    out = out.reset_index(drop=True)

    if len(out) < 4:
        raise ValueError(f"{time_col}有效时间点少于4个。")
    return out


def get_window_indices(time_array, start_time, end_time, label):
    """根据给定时间边界获取区间索引；结束点按包含方式处理。"""
    t = np.asarray(time_array, dtype=float)
    start_time = float(start_time)
    end_time = float(end_time)

    if not np.isfinite(start_time) or not np.isfinite(end_time):
        raise ValueError(f"{label}起止时间不是有效数值。")
    if end_time <= start_time:
        raise ValueError(f"{label}结束时间必须晚于开始时间。")

    idx = np.where((t >= start_time) & (t <= end_time))[0]
    if len(idx) < 4:
        raise ValueError(
            f"{label}窗口内有效采样点不足4个。"
            f"窗口={start_time:.4f}–{end_time:.4f}s，点数={len(idx)}"
        )
    return int(idx[0]), int(idx[-1]) + 1


def find_extreme_event(time_array, signal, start_idx, end_idx, stroke_type, label):
    """
    根据击球类型识别最大骨盆后摆事件：
    forehand -> 最小骨盆旋转角；
    backhand -> 最大骨盆旋转角。
    """
    segment = np.asarray(signal[start_idx:end_idx], dtype=float)
    valid = np.isfinite(segment)

    if np.sum(valid) < 3:
        raise ValueError(f"{label}骨盆旋转事件搜索窗口有效数据不足。")

    valid_positions = np.where(valid)[0]
    valid_values = segment[valid]

    if stroke_type == "forehand":
        local_valid_idx = int(np.argmin(valid_values))
    elif stroke_type == "backhand":
        local_valid_idx = int(np.argmax(valid_values))
    else:
        raise ValueError(
            'STROKE_TYPE只能设置为 "forehand" 或 "backhand"。'
        )

    local_idx = int(valid_positions[local_valid_idx])
    global_idx = start_idx + local_idx

    return {
        "index": global_idx,
        "time": float(time_array[global_idx]),
        "angle": float(signal[global_idx]),
    }


def cubic_resample_1000(signal, target_points=TARGET_POINTS):
    """将单个已截取的动作周期用三次样条插值归一化至1000点。"""
    y = np.asarray(signal, dtype=float)

    if len(y) < 4:
        raise ValueError("动作周期少于4个原始采样点，无法进行三次样条插值。")

    if not np.all(np.isfinite(y)):
        raise ValueError("动作周期中存在NaN/Inf，请先检查原始数据。")

    source_phase = np.linspace(0.0, 1.0, len(y))
    target_phase = np.linspace(0.0, 1.0, target_points)

    spline = interp1d(
        source_phase,
        y,
        kind="cubic",
        bounds_error=True,
    )
    return spline(target_phase)


def calculate_rmse_r(reference, measured):
    """计算RMSE和Pearson r。"""
    ref = np.asarray(reference, dtype=float)
    pred = np.asarray(measured, dtype=float)

    mask = np.isfinite(ref) & np.isfinite(pred)
    if np.sum(mask) < 3:
        return np.nan, np.nan

    ref = ref[mask]
    pred = pred[mask]

    rmse = float(np.sqrt(np.mean((ref - pred) ** 2)))

    if np.std(ref) == 0 or np.std(pred) == 0:
        corr = np.nan
    else:
        corr = float(np.corrcoef(ref, pred)[0, 1])

    return rmse, corr


def interpolate_imu_to_vicon_time(t_imu_aligned, signal_imu, t_vicon):
    """
    将已经完成事件时间对齐的IMU信号映射到Vicon时间戳。
    这一步仅统一两套系统的采样时间点，不是动作周期归一化。
    """
    t_imu_aligned = np.asarray(t_imu_aligned, dtype=float)
    signal_imu = np.asarray(signal_imu, dtype=float)
    t_vicon = np.asarray(t_vicon, dtype=float)

    valid = np.isfinite(t_imu_aligned) & np.isfinite(signal_imu)
    tx = t_imu_aligned[valid]
    yy = signal_imu[valid]

    if len(tx) < 4:
        raise ValueError("IMU有效数据少于4点，无法进行三次样条时间插值。")

    order = np.argsort(tx)
    tx = tx[order]
    yy = yy[order]

    unique_t, unique_idx = np.unique(tx, return_index=True)
    tx = unique_t
    yy = yy[unique_idx]

    if len(tx) < 4:
        raise ValueError("IMU唯一时间戳少于4个。")

    f = interp1d(
        tx,
        yy,
        kind="cubic",
        bounds_error=False,
        fill_value=np.nan,
    )
    return f(t_vicon)


def check_sampling_frequency(time_array, label):
    """打印实际采样频率，核查是否约为100 Hz。"""
    dt = np.diff(np.asarray(time_array, dtype=float))
    dt = dt[np.isfinite(dt) & (dt > 0)]

    if len(dt) == 0:
        print(f"⚠️ {label}无法计算采样频率。")
        return np.nan

    median_dt = float(np.median(dt))
    fs = 1.0 / median_dt if median_dt > 0 else np.nan

    print(f"{label}: median Δt = {median_dt:.6f} s, Fs ≈ {fs:.2f} Hz")
    if np.isfinite(fs) and abs(fs - EXPECTED_FS) > 2:
        warnings.warn(
            f"{label}实际采样频率约为{fs:.2f} Hz，"
            f"与Methods中描述的{EXPECTED_FS:.0f} Hz不一致，请核查。"
        )
    return fs


# ============================================================
# 3. 单个试次：事件对齐、Vicon共同截取、1000点归一化
# ============================================================

def process_one_trial(v_row, i_row):
    action_name = str(v_row["文件名"]).strip()

    vicon_path = find_file_fuzzy(VICON_DIR, action_name, ".mot")
    imu_path = find_file_fuzzy(IMU_DIR, str(i_row["文件名"]).strip(), ".csv")

    if vicon_path is None:
        raise FileNotFoundError(f"找不到Vicon MOT文件：{action_name}")
    if imu_path is None:
        raise FileNotFoundError(f"找不到IMU CSV文件：{i_row['文件名']}")

    df_v = read_vicon_mot(vicon_path)
    df_i = pd.read_csv(imu_path, skipinitialspace=True)
    df_i.columns = [str(c).strip() for c in df_i.columns]

    df_v = clean_sort_time_series(df_v, "time")
    df_i = clean_sort_time_series(df_i, "Time")

    t_v = df_v["time"].to_numpy(dtype=float)
    t_i = df_i["Time"].to_numpy(dtype=float)

    print(f"\n处理试次：{action_name}")
    fs_v = check_sampling_frequency(t_v, "Vicon")
    fs_i = check_sampling_frequency(t_i, "IMU")

    # --------------------------------------------------------
    # A. IMU坐标方向预处理（沿用原分析代码）
    # --------------------------------------------------------
    for joint in IMU_SIGN_FLIP_JOINTS:
        imu_col = smart_get_col(df_i, JOINT_MAPPING[joint]["imu"])
        if imu_col is not None:
            df_i[imu_col] = pd.to_numeric(df_i[imu_col], errors="coerce") * -1

    # --------------------------------------------------------
    # B. 识别两套系统中的最大骨盆后摆事件
    #    这里的时间表窗口仅用于定位同一事件，不作为最终T1–T2截取区间。
    # --------------------------------------------------------
    v_pelvis_col = JOINT_MAPPING["Pelvis Rotation"]["vicon"]
    i_pelvis_col = smart_get_col(
        df_i, JOINT_MAPPING["Pelvis Rotation"]["imu"]
    )

    if v_pelvis_col not in df_v.columns:
        raise KeyError(f"Vicon缺少骨盆旋转列：{v_pelvis_col}")
    if i_pelvis_col is None:
        raise KeyError("IMU缺少Pelvis Rotation列。")

    v_pelvis = pd.to_numeric(
        df_v[v_pelvis_col], errors="coerce"
    ).to_numpy(dtype=float)
    i_pelvis = pd.to_numeric(
        df_i[i_pelvis_col], errors="coerce"
    ).to_numpy(dtype=float)

    # Vicon与IMU各自的粗略事件搜索窗口
    v_search_start, v_search_end = get_window_indices(
        t_v,
        v_row["准备开始点"],
        v_row["随挥结束点"],
        "Vicon事件搜索",
    )
    i_search_start, i_search_end = get_window_indices(
        t_i,
        i_row["准备开始时刻"],
        i_row["随挥结束时刻"],
        "IMU事件搜索",
    )

    v_event = find_extreme_event(
        t_v, v_pelvis,
        v_search_start, v_search_end,
        STROKE_TYPE, "Vicon",
    )
    i_event = find_extreme_event(
        t_i, i_pelvis,
        i_search_start, i_search_end,
        STROKE_TYPE, "IMU",
    )

    # 使IMU事件时间与Vicon事件时间重合
    # t_imu_aligned = t_imu + offset
    time_offset_seconds = v_event["time"] - i_event["time"]
    t_i_aligned = t_i + time_offset_seconds

    print(
        f"Vicon backswing event: {v_event['time']:.6f} s "
        f"({v_event['angle']:.4f}°)"
    )
    print(
        f"IMU backswing event:   {i_event['time']:.6f} s "
        f"({i_event['angle']:.4f}°)"
    )
    print(f"Applied IMU time shift: {time_offset_seconds:+.6f} s")

    # --------------------------------------------------------
    # C. 只用Vicon确定T1、T2；同一窗口应用于Vicon和IMU
    #    假定Vicon时间表中的两列就是由Vicon骨盆旋转轨迹确定的T1/T2。
    # --------------------------------------------------------
    T1 = float(v_row["准备开始点"])
    T2 = float(v_row["随挥结束点"])

    if not np.isfinite(T1) or not np.isfinite(T2) or T2 <= T1:
        raise ValueError("Vicon定义的T1/T2无效。")

    v_cycle_mask = (t_v >= T1) & (t_v <= T2)
    v_cycle_indices = np.where(v_cycle_mask)[0]

    if len(v_cycle_indices) < 4:
        raise ValueError(
            f"Vicon定义的T1–T2区间内采样点不足4个：{T1}–{T2}"
        )

    # 共同分析时间轴就是Vicon在T1–T2内的原始时间戳
    t_v_cycle = t_v[v_cycle_indices]
    phase_1000 = np.linspace(0.0, 100.0, TARGET_POINTS)

    # --------------------------------------------------------
    # D. 同一时间偏移应用于全部IMU关节，并映射至Vicon时间轴
    # --------------------------------------------------------
    normalized_data = {
        "Cycle_Percent": phase_1000,
    }
    trial_metrics = []

    for joint, mapping in JOINT_MAPPING.items():
        v_col = mapping["vicon"]
        i_col = smart_get_col(df_i, mapping["imu"])

        if v_col not in df_v.columns:
            raise KeyError(f"Vicon缺少关节列：{v_col}")
        if i_col is None:
            raise KeyError(f"IMU缺少关节列：{mapping['imu']}")

        v_signal_full = pd.to_numeric(
            df_v[v_col], errors="coerce"
        ).to_numpy(dtype=float)

        i_signal_full = pd.to_numeric(
            df_i[i_col], errors="coerce"
        ).to_numpy(dtype=float)

        # 使用同一个事件偏移量校正所有IMU DOF
        i_signal_on_vicon_clock = interpolate_imu_to_vicon_time(
            t_i_aligned,
            i_signal_full,
            t_v,
        )

        # 同一个Vicon定义的T1–T2窗口截取两套系统
        v_cycle = v_signal_full[v_cycle_indices]
        i_cycle = i_signal_on_vicon_clock[v_cycle_indices]

        # 若区间边界附近出现少量无效插值点，直接报错，避免悄悄改变周期边界
        if not np.all(np.isfinite(v_cycle)):
            raise ValueError(
                f"{joint}: Vicon在T1–T2区间存在NaN/Inf，请检查原始数据。"
            )
        if not np.all(np.isfinite(i_cycle)):
            raise ValueError(
                f"{joint}: IMU在共同Vicon T1–T2区间存在NaN/Inf。"
                "可能是IMU记录覆盖范围不足或时间戳未正确对齐。"
            )

        # 两套系统分别归一化到1000点
        v_1000 = cubic_resample_1000(v_cycle, TARGET_POINTS)
        i_1000 = cubic_resample_1000(i_cycle, TARGET_POINTS)

        rmse, pearson_r = calculate_rmse_r(v_1000, i_1000)

        normalized_data[f"Vicon_{joint}"] = v_1000
        normalized_data[f"IMU_{joint}"] = i_1000

        trial_metrics.append({
            "Action": action_name,
            "Joint": joint,
            "Stroke_Type": STROKE_TYPE,
            "T1_Vicon_s": T1,
            "T2_Vicon_s": T2,
            "Movement_Duration_s": T2 - T1,
            "Vicon_Backswing_Event_s": v_event["time"],
            "IMU_Backswing_Event_s": i_event["time"],
            "IMU_Time_Shift_s": time_offset_seconds,
            "RMSE_deg": rmse,
            "Pearson_r": pearson_r,
            "Vicon_Raw_Samples": len(v_cycle),
            "Normalized_Samples": TARGET_POINTS,
            "Vicon_Fs_Hz": fs_v,
            "IMU_Fs_Hz": fs_i,
        })

    normalized_df = pd.DataFrame(normalized_data)

    event_info = {
        "Action": action_name,
        "Stroke_Type": STROKE_TYPE,
        "Vicon_Event_Time_s": v_event["time"],
        "IMU_Event_Time_s": i_event["time"],
        "IMU_Time_Shift_s": time_offset_seconds,
        "T1_Vicon_s": T1,
        "T2_Vicon_s": T2,
        "Movement_Duration_s": T2 - T1,
        "Vicon_Raw_Cycle_Samples": len(v_cycle_indices),
        "Normalized_Points": TARGET_POINTS,
        "Vicon_Fs_Hz": fs_v,
        "IMU_Fs_Hz": fs_i,
        "Vicon_File": vicon_path,
        "IMU_File": imu_path,
    }

    return normalized_df, trial_metrics, event_info


# ============================================================
# 4. 主程序：匹配试次并批量处理
# ============================================================

def run_analysis():
    if STROKE_TYPE not in ("forehand", "backhand"):
        raise ValueError(
            '请先将STROKE_TYPE设置为 "forehand" 或 "backhand"。'
        )

    if not os.path.isfile(VICON_TIME_FILE):
        raise FileNotFoundError(f"Vicon时间表不存在：{VICON_TIME_FILE}")
    if not os.path.isfile(IMU_TIME_FILE):
        raise FileNotFoundError(f"IMU时间表不存在：{IMU_TIME_FILE}")

    df_v_time = pd.read_excel(VICON_TIME_FILE)
    df_i_time = pd.read_excel(IMU_TIME_FILE)

    required_v = ["文件名", "准备开始点", "随挥结束点"]
    required_i = ["文件名", "准备开始时刻", "随挥结束时刻"]

    missing_v = [c for c in required_v if c not in df_v_time.columns]
    missing_i = [c for c in required_i if c not in df_i_time.columns]

    if missing_v:
        raise KeyError(f"Vicon时间表缺少列：{missing_v}")
    if missing_i:
        raise KeyError(f"IMU时间表缺少列：{missing_i}")

    all_metrics = []
    all_events = []
    successful_trials = 0

    normalized_dir = os.path.join(SAVE_DIR, "Normalized_1000_Point_Trajectories")
    os.makedirs(normalized_dir, exist_ok=True)

    for _, v_row in df_v_time.iterrows():
        action_name = str(v_row["文件名"]).strip()
        v_id = standardize_id(action_name)

        matched = df_i_time[
            df_i_time["文件名"].apply(standardize_id) == v_id
        ]

        if matched.empty:
            print(f"⚠️ 找不到对应IMU时间表记录：{action_name}")
            continue

        i_row = matched.iloc[0]

        try:
            normalized_df, trial_metrics, event_info = process_one_trial(
                v_row, i_row
            )

            out_name = re.sub(r'[\\/:*?"<>|]+', "_", action_name)
            trajectory_path = os.path.join(
                normalized_dir,
                f"{out_name}_Aligned_CommonWindow_1000points.csv",
            )
            normalized_df.to_csv(
                trajectory_path,
                index=False,
                encoding="utf-8-sig",
            )

            all_metrics.extend(trial_metrics)
            all_events.append(event_info)
            successful_trials += 1

            print(
                f"✅ 完成 {action_name}: "
                f"offset={event_info['IMU_Time_Shift_s']:+.5f}s, "
                f"T1–T2={event_info['Movement_Duration_s']:.4f}s, "
                f"输出1000点"
            )

        except Exception as exc:
            print(f"❌ 试次处理失败：{action_name} | {exc}")

    if successful_trials == 0:
        print(
            "\n没有成功处理任何试次。请检查：动作类型、文件名匹配、"
            "时间表时间单位、T1/T2定义以及原始数据时间戳。"
        )
        return

    metrics_df = pd.DataFrame(all_metrics)
    events_df = pd.DataFrame(all_events)

    summary_df = (
        metrics_df.groupby("Joint")[["RMSE_deg", "Pearson_r"]]
        .agg(["mean", "std", "count"])
    )

    excel_path = os.path.join(
        SAVE_DIR,
        "Aligned_CommonSegmentation_1000Point_Analysis.xlsx",
    )

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        metrics_df.to_excel(
            writer, sheet_name="Trial_Level_Metrics", index=False
        )
        events_df.to_excel(
            writer, sheet_name="Synchronization_and_T1T2", index=False
        )
        summary_df.to_excel(
            writer, sheet_name="Group_Summary"
        )

    print("\n" + "=" * 72)
    print("全部处理完成")
    print(f"成功试次数：{successful_trials}")
    print(f"归一化轨迹目录：{normalized_dir}")
    print(f"指标汇总文件：{excel_path}")
    print("每个试次输出：11个DOF、Vicon/IMU两套曲线、每套1000个点。")
    print("=" * 72)


if __name__ == "__main__":
    run_analysis()
