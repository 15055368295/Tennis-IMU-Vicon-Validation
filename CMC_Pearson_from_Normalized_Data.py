
from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
from scipy.stats import t as student_t


# ============================================================
# 1. 用户设置：输入文件夹和输出文件
# ============================================================
NORMALIZED_DIRS = {
    "GZ": Path(r"E:\RMSE分析结果\GZ_Aligned_1000Points\Normalized_1000_Point_Trajectories"),
    "KZ": Path(r"E:\RMSE分析结果\KZ_Aligned_1000Points\Normalized_1000_Point_Trajectories"),
    "GF": Path(r"E:\RMSE分析结果\GF_Aligned_1000Points\Normalized_1000_Point_Trajectories"),
}

OUTPUT_DIR = Path(r"E:\CMC_Pearson分析结果\Participant_Level_CMC_Pearson")
OUTPUT_FILE = "CMC_Pearson_Participant_Level_Analysis.xlsx"

CONDITIONS = ["GZ", "KZ", "GF"]
EXPECTED_N_SUBJECTS = 16
TARGET_POINTS = 1000

# 文件命名格式示例：ZA01_XYR_GZ01
# ZA01：类别/性别等分组代码，不作为受试者唯一编号
# XYR：受试者姓名缩写，作为受试者唯一标识
# GZ：动作条件代码
# 01：该动作条件下的试次序号
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
# 2. 文件名识别与数据检查
# ============================================================

def action_name_from_file(csv_path: Path) -> str:
    """从归一化CSV文件名中去掉固定后缀，得到原始试次名称。"""
    stem = csv_path.stem
    if stem.endswith(FILE_SUFFIX):
        stem = stem[:-len(FILE_SUFFIX)]
    return stem


def extract_subject_id(action_name: str) -> str:
    """从类似ZA01_XYR_GZ01的文件名中提取受试者缩写XYR。"""
    if action_name in SUBJECT_OVERRIDES:
        return str(SUBJECT_OVERRIDES[action_name]).strip().upper()

    match = re.search(SUBJECT_REGEX, action_name)
    if not match:
        raise ValueError(
            f"无法从试次文件名提取受试者缩写：{action_name!r}。"
            "预期格式类似ZA01_XYR_GZ01，请检查文件名或设置SUBJECT_OVERRIDES。"
        )

    return match.group(1).upper()


def validate_paired_trajectories(vicon_values, imu_values, file_name, joint):
    """
    检查Vicon和IMU两条已归一化曲线。
    不删除缺失点、不进行再次归一化；数据不完整时直接报错。
    """
    vicon = np.asarray(vicon_values, dtype=float)
    imu = np.asarray(imu_values, dtype=float)

    if vicon.ndim != 1 or imu.ndim != 1:
        raise ValueError(f"{file_name} | {joint}：输入数据必须是一维数组。")

    if vicon.shape != imu.shape:
        raise ValueError(
            f"{file_name} | {joint}：Vicon和IMU长度不一致："
            f"{len(vicon)} vs {len(imu)}。"
        )

    if len(vicon) != TARGET_POINTS:
        raise ValueError(
            f"{file_name} | {joint}：当前为{len(vicon)}点，"
            f"而不是要求的{TARGET_POINTS}点。"
        )

    if not np.all(np.isfinite(vicon)) or not np.all(np.isfinite(imu)):
        raise ValueError(
            f"{file_name} | {joint}：数据包含NaN或Inf。"
            "本程序不会自动删除缺失点，请检查归一化CSV。"
        )

    return vicon, imu


# ============================================================
# 3. 试次级CMC和Pearson r计算
# ============================================================

def calculate_cmc(vicon_values, imu_values) -> float:
    """
    根据两条配对运动学轨迹计算CMC。

    G：测量系统数量（Vicon和IMU，因此G=2）
    N：归一化时间点数量（本研究为1000）
    """
    vicon = np.asarray(vicon_values, dtype=float)
    imu = np.asarray(imu_values, dtype=float)

    if vicon.shape != imu.shape:
        raise ValueError("计算CMC时，Vicon和IMU曲线长度必须一致。")

    if vicon.ndim != 1 or len(vicon) != TARGET_POINTS:
        raise ValueError(f"CMC计算要求输入{TARGET_POINTS}个归一化点。")

    if not np.all(np.isfinite(vicon)) or not np.all(np.isfinite(imu)):
        raise ValueError("CMC输入曲线存在NaN或Inf，无法计算。")

    data = np.vstack([vicon, imu])
    G, N = data.shape

    # 所有系统、所有时间点的总均值
    grand_mean = np.mean(data)

    # 每个归一化时间点上，两个系统的平均值
    time_point_mean = np.mean(data, axis=0)

    # 总平方和与误差平方和
    ss_total = np.sum((data - grand_mean) ** 2)
    ss_error = np.sum((data - time_point_mean) ** 2)

    # 总变异为0时，CMC无法定义
    if ss_total <= 0:
        return np.nan

    # 均方误差与总均方
    ms_error = ss_error / (N * (G - 1))
    ms_total = ss_total / (N * G - 1)

    if ms_total <= 0:
        return np.nan

    cmc_squared = 1.0 - (ms_error / ms_total)

    # 与原分析公式一致：若数值结果略低于0，则将CMC平方项截为0，
    # 避免对负数开平方。
    return float(np.sqrt(max(0.0, cmc_squared)))


def calculate_pearson_r(vicon_values, imu_values) -> float:
    """计算两条1000点归一化运动学轨迹之间的Pearson相关系数。"""
    vicon = np.asarray(vicon_values, dtype=float)
    imu = np.asarray(imu_values, dtype=float)

    if vicon.shape != imu.shape:
        raise ValueError("计算Pearson r时，Vicon和IMU曲线长度必须一致。")

    if vicon.ndim != 1 or len(vicon) != TARGET_POINTS:
        raise ValueError(f"Pearson r计算要求输入{TARGET_POINTS}个归一化点。")

    if not np.all(np.isfinite(vicon)) or not np.all(np.isfinite(imu)):
        raise ValueError("Pearson r输入曲线存在NaN或Inf，无法计算。")

    # 任一曲线为常数时，相关系数无法定义
    if np.std(vicon) == 0 or np.std(imu) == 0:
        return np.nan

    return float(np.corrcoef(vicon, imu)[0, 1])


# ============================================================
# 4. 读取归一化CSV并计算每个试次的CMC和Pearson r
# ============================================================

def load_trial_level_metrics() -> pd.DataFrame:
    records = []

    for condition in CONDITIONS:
        folder = NORMALIZED_DIRS[condition]

        if not folder.exists():
            raise FileNotFoundError(
                f"{condition}归一化数据目录不存在：{folder}\n"
                "请检查NORMALIZED_DIRS中的路径。"
            )

        csv_files = sorted(
            folder.glob("*_Aligned_CommonWindow_1000points.csv")
        )

        if not csv_files:
            raise FileNotFoundError(
                f"{folder}中没有找到归一化CSV文件。\n"
                "预期文件名后缀为_Aligned_CommonWindow_1000points.csv。"
            )

        print(f"\n读取条件{condition}：找到{len(csv_files)}个归一化试次")

        for csv_path in csv_files:
            action = action_name_from_file(csv_path)
            subject = extract_subject_id(action)

            df = pd.read_csv(csv_path)

            if len(df) != TARGET_POINTS:
                raise ValueError(
                    f"{csv_path.name}包含{len(df)}行，"
                    f"不是要求的{TARGET_POINTS}行。"
                )

            for joint in JOINTS:
                vicon_col = f"Vicon_{joint}"
                imu_col = f"IMU_{joint}"

                missing_columns = [
                    col for col in (vicon_col, imu_col)
                    if col not in df.columns
                ]
                if missing_columns:
                    raise KeyError(
                        f"文件{csv_path.name}缺少列：{missing_columns}。"
                        "请确认列名与第一段归一化程序一致。"
                    )

                vicon, imu = validate_paired_trajectories(
                    pd.to_numeric(df[vicon_col], errors="coerce").to_numpy(),
                    pd.to_numeric(df[imu_col], errors="coerce").to_numpy(),
                    csv_path.name,
                    joint,
                )

                cmc_value = calculate_cmc(vicon, imu)
                pearson_value = calculate_pearson_r(vicon, imu)

                records.append({
                    "Subject": subject,
                    "Stroke_Condition": condition,
                    "Action": action,
                    "Joint": joint,
                    "CMC": cmc_value,
                    "Pearson_R": pearson_value,
                    "N_Normalized_Points": TARGET_POINTS,
                    "Source_CSV": csv_path.name,
                })

            print(f"  已计算：{action} | 受试者={subject}")

    trial_df = pd.DataFrame(records)

    if trial_df.empty:
        raise RuntimeError("没有计算出任何试次级CMC和Pearson r。")

    # 检查同一受试者、条件、试次和DOF是否出现重复记录
    duplicate_key = ["Subject", "Stroke_Condition", "Action", "Joint"]
    duplicated = trial_df.duplicated(duplicate_key, keep=False)

    if duplicated.any():
        examples = trial_df.loc[duplicated, duplicate_key].head(10)
        raise ValueError(
            "发现重复的受试者/动作条件/试次/关节记录，请检查归一化文件：\n"
            + examples.to_string(index=False)
        )

    return trial_df


# ============================================================
# 5. 受试者水平汇总及95%置信区间
# ============================================================

def summarize_participant_values(
    data: pd.DataFrame,
    value_col: str,
    group_columns: list,
    metric_name: str,
) -> pd.DataFrame:
    """以受试者为统计单位，计算Mean、样本SD和双侧95% Student-t CI。"""
    rows = []

    for keys, group in data.groupby(group_columns, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        values = (
            pd.to_numeric(group[value_col], errors="coerce")
            .dropna()
            .to_numpy(dtype=float)
        )

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
            "Metric": metric_name,
            "N_Participants": n,
            f"Mean_{metric_name}": mean_value,
            f"SD_{metric_name}": sd_value,
            f"CI95_Lower_{metric_name}": ci_low,
            f"CI95_Upper_{metric_name}": ci_high,
        })
        rows.append(row)

    return pd.DataFrame(rows)


def aggregate_results(trial_df: pd.DataFrame):
    # 第一步：同一受试者、同一动作条件、同一DOF内，
    # 对所有重复试次的CMC和Pearson r分别取算术平均。
    participant_condition = (
        trial_df.groupby(
            ["Subject", "Stroke_Condition", "Joint"],
            as_index=False,
        )
        .agg(
            CMC=("CMC", "mean"),
            Pearson_R=("Pearson_R", "mean"),
            N_Trials=("Action", "nunique"),
        )
    )

    # 第二步：对每名受试者、每个DOF的GZ、KZ、GF结果等权平均。
    # 必须有三个条件的结果；缺少任何一个条件时，Overall留空。
    participant_overall = None
    overall_columns = []

    for metric in ["CMC", "Pearson_R"]:
        wide = participant_condition.pivot(
            index=["Subject", "Joint"],
            columns="Stroke_Condition",
            values=metric,
        )
        wide = wide.reindex(columns=CONDITIONS)

        n_conditions = wide.notna().sum(axis=1)
        overall_values = wide.mean(axis=1, skipna=True).where(
            n_conditions == len(CONDITIONS)
        )

        temp = wide.reset_index()
        temp[f"Overall_{metric}"] = overall_values.to_numpy()
        temp[f"N_Conditions_Available_{metric}"] = n_conditions.to_numpy()

        if participant_overall is None:
            participant_overall = temp
        else:
            participant_overall = participant_overall.merge(
                temp[["Subject", "Joint", f"Overall_{metric}",
                      f"N_Conditions_Available_{metric}"]],
                on=["Subject", "Joint"],
                how="outer",
            )

        overall_columns.append(f"Overall_{metric}")

    # 第三步：按每个动作条件，使用受试者水平数据计算组统计。
    cmc_condition_summary = summarize_participant_values(
        participant_condition,
        value_col="CMC",
        group_columns=["Stroke_Condition", "Joint"],
        metric_name="CMC",
    )

    pearson_condition_summary = summarize_participant_values(
        participant_condition,
        value_col="Pearson_R",
        group_columns=["Stroke_Condition", "Joint"],
        metric_name="Pearson_R",
    )

    # 第四步：Overall统计基于每名受试者三个动作条件等权平均后的数值。
    cmc_overall_summary = summarize_participant_values(
        participant_overall,
        value_col="Overall_CMC",
        group_columns=["Joint"],
        metric_name="CMC",
    )
    cmc_overall_summary.insert(0, "Stroke_Condition", "Overall")

    pearson_overall_summary = summarize_participant_values(
        participant_overall,
        value_col="Overall_Pearson_R",
        group_columns=["Joint"],
        metric_name="Pearson_R",
    )
    pearson_overall_summary.insert(0, "Stroke_Condition", "Overall")

    cmc_summary = pd.concat(
        [cmc_condition_summary, cmc_overall_summary],
        ignore_index=True,
        sort=False,
    )
    pearson_summary = pd.concat(
        [pearson_condition_summary, pearson_overall_summary],
        ignore_index=True,
        sort=False,
    )

    # 受试者数量和动作条件完整性检查：只提示，不剔除任何试次。
    all_subjects = sorted(trial_df["Subject"].dropna().unique().tolist())

    if len(all_subjects) != EXPECTED_N_SUBJECTS:
        warnings.warn(
            f"当前识别到{len(all_subjects)}名受试者，"
            f"{EXPECTED_N_SUBJECTS}核对文件名及输入数据。"
        )

    incomplete_cmc = participant_overall[
        participant_overall["N_Conditions_Available_CMC"] != len(CONDITIONS)
    ]
    incomplete_pearson = participant_overall[
        participant_overall["N_Conditions_Available_Pearson_R"] != len(CONDITIONS)
    ]

    if not incomplete_cmc.empty or not incomplete_pearson.empty:
        warnings.warn(
            "部分受试者/DOF缺少GZ、KZ、GF中的一个或多个条件。"
            "对应Overall结果留空，不使用不完整条件计算。"
        )

    # 每名受试者、每个动作条件实际纳入的重复试次数量
    trial_counts = (
        trial_df.groupby(
            ["Subject", "Stroke_Condition"],
            as_index=False,
        )
        .agg(N_Trials=("Action", "nunique"))
    )

    return (
        participant_condition,
        participant_overall,
        cmc_summary,
        pearson_summary,
        trial_counts,
    )


# ============================================================
# 6. 主程序：输出Excel结果
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trial_df = load_trial_level_metrics()

    (
        participant_condition,
        participant_overall,
        cmc_summary,
        pearson_summary,
        trial_counts,
    ) = aggregate_results(trial_df)

    output_path = OUTPUT_DIR / OUTPUT_FILE

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        trial_df.to_excel(
            writer,
            sheet_name="Trial_Level_CMC_Pearson",
            index=False,
        )
        participant_condition.to_excel(
            writer,
            sheet_name="Participant_Condition_Mean",
            index=False,
        )
        participant_overall.to_excel(
            writer,
            sheet_name="Participant_Overall_Mean",
            index=False,
        )
        cmc_summary.to_excel(
            writer,
            sheet_name="Group_Summary_CMC",
            index=False,
        )
        pearson_summary.to_excel(
            writer,
            sheet_name="Group_Summary_Pearson",
            index=False,
        )
        trial_counts.to_excel(
            writer,
            sheet_name="Trial_Count_QC",
            index=False,
        )

    print("\n" + "=" * 78)
    print("分析完成")
    print(f"识别受试者数量：{trial_df['Subject'].nunique()}（论文预期：{EXPECTED_N_SUBJECTS}）")
    print(f"试次数量：{trial_df[['Stroke_Condition', 'Action']].drop_duplicates().shape[0]}")
    print(f"试次级CMC/Pearson记录数：{len(trial_df)}")
    print(f"结果文件：{output_path}")

    print("\nOverall CMC汇总：")
    overall_cmc = cmc_summary[cmc_summary["Stroke_Condition"] == "Overall"]
    print(
        overall_cmc[
            ["Joint", "N_Participants", "Mean_CMC", "SD_CMC",
             "CI95_Lower_CMC", "CI95_Upper_CMC"]
        ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )

    print("\nOverall Pearson r汇总：")
    overall_pearson = pearson_summary[
        pearson_summary["Stroke_Condition"] == "Overall"
    ]
    print(
        overall_pearson[
            ["Joint", "N_Participants", "Mean_Pearson_R", "SD_Pearson_R",
             "CI95_Lower_Pearson_R", "CI95_Upper_Pearson_R"]
        ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )

    print("=" * 78)


if __name__ == "__main__":
    main()
