
from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
from scipy.stats import t as student_t


# ============================================================
# 1. 用户设置：文件路径
# ============================================================

NORMALIZED_DIRS = {
    "GZ": Path(r"E:\RMSE分析结果\GZ_Aligned_1000Points\Normalized_1000_Point_Trajectories"),
    "KZ": Path(r"E:\RMSE分析结果\KZ_Aligned_1000Points\Normalized_1000_Point_Trajectories"),
    "GF": Path(r"E:\RMSE分析结果\GF_Aligned_1000Points\Normalized_1000_Point_Trajectories"),
}

OUTPUT_DIR = Path(r"E:\RMSE分析结果\Final_Participant_Level_RMSE")
OUTPUT_FILE = "RMSE_Participant_Level_Analysis.xlsx"

CONDITIONS = ["GZ", "KZ", "GF"]
EXPECTED_N_SUBJECTS = 16
TARGET_POINTS = 1000

#命名格式为：ZA01_XYR_GZ01
# 各字段含义：
#   ZA01：受试者类别/性别等分组代码，不作为受试者唯一编号；
#   XYR ：受试者姓名缩写，是本程序用于区分受试者的唯一标识；
#   GZ  ：动作/击球条件代码（GZ、KZ或GF）；
#   01  ：该受试者在该动作条件下的试次序号，不作为受试者编号。
SUBJECT_REGEX = r"(?i)(?:^|[_\s-])([A-Z]{2,5})(?=[_\s-](?:GZ|KZ|GF)\d+)"

SUBJECT_OVERRIDES = {}

FILE_SUFFIX = "_Aligned_CommonWindow_1000points"

JOINTS = [
    "Pelvis Tilt",
    "Pelvis Obliquity",
    "Pelvis Rotation",
    "R-Hip Flex/Ext",
    "R-Hip Addu/Abdu",
    "R-Hip Rotation",
    "R-Knee Flex/Ext",
    "L-Hip Flex/Ext",
    "L-Hip Addu/Abdu",
    "L-Hip Rotation",
    "L-Knee Flex/Ext",
]


# ============================================================
# 2. 辅助函数
# ============================================================

def action_name_from_file(csv_path: Path) -> str:
    """从归一化CSV文件名中还原原始Action/试次名称。"""
    stem = csv_path.stem
    if stem.endswith(FILE_SUFFIX):
        stem = stem[:-len(FILE_SUFFIX)]
    return stem


def extract_subject_id(action_name: str) -> str:
    """从文件名中提取受试者姓名缩写（如XYR），不使用动作代码或试次序号。"""
    if action_name in SUBJECT_OVERRIDES:
        return str(SUBJECT_OVERRIDES[action_name]).strip().upper()

    match = re.search(SUBJECT_REGEX, action_name)
    if not match:
        raise ValueError(
            f"无法从试次文件名提取受试者缩写：{action_name!r}。"
            "预期格式类似 ZA01_XYR_GZ01；请检查分隔符和动作代码，"
            "或在 SUBJECT_OVERRIDES 中添加该试次的手动映射。"
        )

    # 返回姓名缩写作为受试者唯一标识，例如 XYR。
    return match.group(1).upper()


def calculate_trial_rmse(vicon_values, imu_values) -> tuple[float, int]:
    """计算两条已经完成归一化的配对运动学轨迹之间的RMSE。"""
    vicon = np.asarray(vicon_values, dtype=float)
    imu = np.asarray(imu_values, dtype=float)

    if vicon.shape != imu.shape:
        raise ValueError(
            f"Vicon/IMU曲线长度不一致：{len(vicon)} vs {len(imu)}。"
        )

    if len(vicon) != TARGET_POINTS:
        raise ValueError(
            f"归一化点数不是{TARGET_POINTS}：当前为{len(vicon)}。"
            "请检查输入CSV是否来自第一段1000点归一化程序。"
        )

    valid = np.isfinite(vicon) & np.isfinite(imu)
    n_valid = int(valid.sum())
    if n_valid != TARGET_POINTS:
        raise ValueError(
            f"配对轨迹存在NaN/Inf：有效点数={n_valid}/{TARGET_POINTS}。"
            "为保持每个试次均基于完整1000点计算，本程序不自动删除缺失点。"
        )

    rmse = float(np.sqrt(np.mean((vicon - imu) ** 2)))
    return rmse, n_valid


def summarize_participant_values(
    data: pd.DataFrame,
    value_col: str,
    group_columns: list[str],
    summary_label: str,
) -> pd.DataFrame:
    """基于受试者数据计算均值、样本标准差及双侧95% Student-t置信区间。"""
    rows = []
    grouped = data.groupby(group_columns, dropna=False)

    for keys, group in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)

        values = pd.to_numeric(group[value_col], errors="coerce").dropna().to_numpy(float)
        n = len(values)
        mean_value = float(np.mean(values)) if n else np.nan
        sd_value = float(np.std(values, ddof=1)) if n >= 2 else np.nan

        if n >= 2:
            t_critical = float(student_t.ppf(0.975, df=n - 1))
            margin = t_critical * sd_value / np.sqrt(n)
            ci_low = mean_value - margin
            ci_high = mean_value + margin
        else:
            ci_low = np.nan
            ci_high = np.nan

        row = dict(zip(group_columns, keys))
        row.update({
            "N_Participants": n,
            "Mean_RMSE_deg": mean_value,
            "SD_RMSE_deg": sd_value,
            "CI95_Lower_deg": ci_low,
            "CI95_Upper_deg": ci_high,
            "Summary": summary_label,
        })
        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# 3. 读取归一化CSV文件并计算试次级RMSE
# ============================================================

def load_trial_level_rmse() -> pd.DataFrame:
    records = []

    for condition in CONDITIONS:
        folder = NORMALIZED_DIRS[condition]
        if not folder.exists():
            raise FileNotFoundError(
                f"{condition}归一化数据目录不存在：{folder}\n"
                "请检查 NORMALIZED_DIRS 中的路径。"
            )

        csv_files = sorted(folder.glob("*_Aligned_CommonWindow_1000points.csv"))
        if not csv_files:
            raise FileNotFoundError(
                f"目录中没有找到归一化CSV：{folder}\n"
                "预期文件名后缀为 _Aligned_CommonWindow_1000points.csv。"
            )

        print(f"\n读取条件 {condition}：{len(csv_files)} 个归一化试次")

        for csv_path in csv_files:
            action = action_name_from_file(csv_path)
            subject = extract_subject_id(action)
            df = pd.read_csv(csv_path)

            if len(df) != TARGET_POINTS:
                raise ValueError(
                    f"{csv_path.name} 行数为{len(df)}，不是{TARGET_POINTS}。"
                )

            for joint in JOINTS:
                vicon_col = f"Vicon_{joint}"
                imu_col = f"IMU_{joint}"

                missing = [c for c in (vicon_col, imu_col) if c not in df.columns]
                if missing:
                    raise KeyError(
                        f"文件 {csv_path.name} 缺少列：{missing}。"
                        "请确认列名与第一段归一化程序一致。"
                    )

                rmse, n_points = calculate_trial_rmse(
                    pd.to_numeric(df[vicon_col], errors="coerce").to_numpy(),
                    pd.to_numeric(df[imu_col], errors="coerce").to_numpy(),
                )

                records.append({
                    "Subject": subject,
                    "Stroke_Condition": condition,
                    "Action": action,
                    "Joint": joint,
                    "RMSE_deg": rmse,
                    "N_Normalized_Points": n_points,
                    "Source_CSV": csv_path.name,
                })

            print(f"  已计算：{action} | Subject={subject}")

    trial_df = pd.DataFrame(records)
    if trial_df.empty:
        raise RuntimeError("没有计算出任何试次级RMSE。")

    duplicate_key = ["Subject", "Stroke_Condition", "Action", "Joint"]
    duplicated = trial_df.duplicated(duplicate_key, keep=False)
    if duplicated.any():
        examples = trial_df.loc[duplicated, duplicate_key].head(10)
        raise ValueError(
            "发现重复的受试者/条件/试次/关节记录，请检查CSV文件：\n"
            + examples.to_string(index=False)
        )

    return trial_df


# ============================================================
# 4. 结果汇总：试次级 -> 受试者/动作条件级 -> 总体
# ============================================================

def aggregate_results(trial_df: pd.DataFrame):
    # 第一步：在每名受试者、每种击球条件及每个DOF内，
    # 对该受试者的重复试次级RMSE求平均。
    participant_condition = (
        trial_df.groupby(
            ["Subject", "Stroke_Condition", "Joint"], as_index=False
        )
        .agg(
            RMSE_deg=("RMSE_deg", "mean"),
            N_Retained_Trials=("Action", "nunique"),
        )
    )

    # 第二步：对每名受试者、每个DOF的GZ、KZ和GF三个条件结果进行等权平均。
    # 必须具备全部三个条件；如果缺少条件，不自动使用仅有的一个或两个条件计算。
    wide = participant_condition.pivot(
        index=["Subject", "Joint"],
        columns="Stroke_Condition",
        values="RMSE_deg",
    )
    wide = wide.reindex(columns=CONDITIONS)

    n_conditions = wide.notna().sum(axis=1)
    overall_values = wide.mean(axis=1, skipna=True).where(n_conditions == len(CONDITIONS))

    participant_overall = wide.reset_index()
    participant_overall["N_Conditions_Available"] = n_conditions.to_numpy()
    participant_overall["Overall_RMSE_deg"] = overall_values.to_numpy()

    # 第三步：组水平统计基于受试者级结果计算
    condition_summary = summarize_participant_values(
        participant_condition,
        value_col="RMSE_deg",
        group_columns=["Stroke_Condition", "Joint"],
        summary_label="Participant mean within stroke condition",
    )

    overall_summary = summarize_participant_values(
        participant_overall,
        value_col="Overall_RMSE_deg",
        group_columns=["Joint"],
        summary_label="Equal-weighted mean of GZ, KZ, and GF within participant",
    )
    overall_summary.insert(0, "Stroke_Condition", "Overall")

    group_summary = pd.concat(
        [condition_summary, overall_summary],
        ignore_index=True,
        sort=False,
    )

    all_subjects = sorted(trial_df["Subject"].dropna().unique().tolist())
    if len(all_subjects) != EXPECTED_N_SUBJECTS:
        warnings.warn(
            f"当前识别到{len(all_subjects)}名受试者，而论文描述为{EXPECTED_N_SUBJECTS}名。"
            "请检查文件名中的受试者编号提取规则及各条件数据是否齐全。"
        )

    incomplete = participant_overall.loc[
        participant_overall["N_Conditions_Available"] != len(CONDITIONS)
    ]
    if not incomplete.empty:
        warnings.warn(
            "部分受试者/关节缺少GZ、KZ、GF中的一个或多个条件；"
            "其Overall_RMSE_deg已留空，不会用不完整条件计算等权总体值。"
        )

    trial_counts = (
        trial_df.groupby(["Subject", "Stroke_Condition", "Joint"], as_index=False)
        .agg(N_Retained_Trials=("Action", "nunique"))
    )

    return participant_condition, participant_overall, group_summary, trial_counts


# ============================================================
# 5. 主程序
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 76)
    print("RMSE分析：直接读取已归一化的1000点Vicon/IMU轨迹")
    print("不重新对齐、不重新截取、不重新插值、不重新归一化、不做IQR试次剔除")
    print("=" * 76)

    trial_df = load_trial_level_rmse()
    participant_condition, participant_overall, group_summary, trial_counts = (
        aggregate_results(trial_df)
    )

    output_path = OUTPUT_DIR / OUTPUT_FILE
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        trial_df.to_excel(writer, sheet_name="Trial_Level_RMSE", index=False)
        participant_condition.to_excel(
            writer, sheet_name="Participant_Condition_Mean", index=False
        )
        participant_overall.to_excel(
            writer, sheet_name="Participant_Overall_Mean", index=False
        )
        group_summary.to_excel(writer, sheet_name="Group_Summary", index=False)
        trial_counts.to_excel(writer, sheet_name="Trial_Count_QC", index=False)

    print("\n" + "=" * 76)
    print("分析完成")
    print(f"识别受试者数量：{trial_df['Subject'].nunique()}（论文预期：{EXPECTED_N_SUBJECTS}）")
    print(f"纳入试次数量：{trial_df[['Stroke_Condition', 'Action']].drop_duplicates().shape[0]}")
    print(f"试次级RMSE记录数：{len(trial_df)}")
    print(f"结果文件：{output_path}")
    print("\nGroup_Summary中的Mean_RMSE_deg和SD_RMSE_deg为最终组水平统计；")
    print("Overall行基于每名受试者三个条件等权平均后的participant-level数值。")
    print("=" * 76)

    print("\n总体RMSE汇总（按关节）：")
    overall_display = group_summary[group_summary["Stroke_Condition"] == "Overall"]
    print(
        overall_display[
            ["Joint", "N_Participants", "Mean_RMSE_deg", "SD_RMSE_deg", "CI95_Lower_deg", "CI95_Upper_deg"]
        ].to_string(index=False, float_format=lambda x: f"{x:.3f}")
    )


if __name__ == "__main__":
    main()
