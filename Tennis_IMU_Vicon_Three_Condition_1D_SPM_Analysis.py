from pathlib import Path
from importlib.metadata import version, PackageNotFoundError
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import spm1d


# ============================================================
# 1. 用户参数设置
# ============================================================

# SAVE_DIR / Normalized_1000_Point_Trajectories
NORMALIZED_DIRS = {
    "GZ": Path(
        r"E:\RMSE分析结果\GZ_Aligned_1000Points\Normalized_1000_Point_Trajectories"
    ),
    "KZ": Path(
        r"E:\RMSE分析结果\KZ_Aligned_1000Points\Normalized_1000_Point_Trajectories"
    ),
    "GF": Path(
        r"E:\RMSE分析结果\GF_Aligned_1000Points\Normalized_1000_Point_Trajectories"
    ),
}

OUTPUT_DIR = Path(r"E:\RMSE分析结果\SPM_All_Conditions_Analysis")
OUTPUT_EXCEL = "SPM_All_Conditions_Participant_Level_Results.xlsx"

CONDITIONS = ["GZ", "KZ", "GF"]
EXPECTED_N_SUBJECTS = 16
TARGET_POINTS = 1000

# 对全部33项比较进行Bonferroni校正：
# 11个DOF × 3种动作条件。
FAMILYWISE_ALPHA = 0.05
N_PLANNED_COMPARISONS = 33
ALPHA_ADJUSTED = FAMILYWISE_ALPHA / N_PLANNED_COMPARISONS

FILE_SUFFIX = "_Aligned_CommonWindow_1000points"

# 文件名示例：ZA01_XYR_GZ01
# 其中XYR为受试者标识，GZ/KZ/GF为动作条件。
# 同时兼容GZ_01、KZ_01、GF_01等写法。
SUBJECT_REGEX = (
    r"(?i)(?:^|[_\s-])([A-Z]{2,5})"
    r"(?=[_\s-](?:GZ|KZ|GF)[_\s-]?\d+)"
)

# 如果个别文件名无法自动识别受试者编号，可在这里手动指定。
# 键为去掉FILE_SUFFIX后的完整试次文件名，值为受试者标识。
# 示例：{"特殊文件名": "XYR"}
SUBJECT_OVERRIDES = {}

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
ANONYMIZE_OUTPUT_IDS = True

# 图片格式设置。
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.linewidth"] = 1.2
plt.rcParams["xtick.labelsize"] = 9
plt.rcParams["ytick.labelsize"] = 9


# ============================================================
# 2. 文件名识别与数据检查
# ============================================================

def get_action_name(csv_path: Path) -> str:
    """从CSV文件名中去除固定的归一化数据后缀。"""
    stem = csv_path.stem
    if stem.lower().endswith(FILE_SUFFIX.lower()):
        return stem[:-len(FILE_SUFFIX)]
    return stem


def extract_condition(action_name: str) -> str:
    """从试次文件名中识别GZ、KZ或GF动作条件。"""
    matches = re.findall(
        r"(?i)(?:^|[_\s-])(GZ|KZ|GF)[_\s-]?\d+",
        action_name
    )
    if len(matches) != 1:
        raise ValueError(
            f"无法唯一识别动作条件：{action_name}。"
            "文件名应包含且只包含一个GZ、KZ或GF标识。"
        )
    return matches[0].upper()


def extract_subject_id(action_name: str) -> str:
    """从试次文件名中提取受试者标识。"""
    if action_name in SUBJECT_OVERRIDES:
        return str(SUBJECT_OVERRIDES[action_name]).strip().upper()

    match = re.search(SUBJECT_REGEX, action_name)
    if not match:
        raise ValueError(
            f"无法从文件名中提取受试者编号：{action_name!r}。"
            "预期格式示例：ZA01_XYR_GZ01。"
            "请检查SUBJECT_REGEX，或在SUBJECT_OVERRIDES中添加映射。"
        )
    return match.group(1).upper()


def validate_trajectory(vicon_values, imu_values, file_name: str, joint: str):
    """
    检查一组已经归一化的Vicon/IMU配对轨迹。
    本函数不会删除数据点，也不会改变轨迹。
    """
    vicon = np.asarray(vicon_values, dtype=float)
    imu = np.asarray(imu_values, dtype=float)

    if vicon.ndim != 1 or imu.ndim != 1:
        raise ValueError(f"{file_name} | {joint}：轨迹必须是一维数组。")

    if vicon.shape != imu.shape:
        raise ValueError(
            f"{file_name} | {joint}：Vicon与IMU长度不一致"
            f"（{len(vicon)} vs {len(imu)}）。"
        )

    if len(vicon) != TARGET_POINTS:
        raise ValueError(
            f"{file_name} | {joint}：应有{TARGET_POINTS}个归一化点，"
            f"实际为{len(vicon)}个。"
        )

    if not np.all(np.isfinite(vicon)) or not np.all(np.isfinite(imu)):
        raise ValueError(
            f"{file_name} | {joint}：检测到NaN或Inf。"
            "程序不会自动删除这些点，请检查前处理输出文件。"
        )

    return vicon, imu


# ============================================================
# 3. 读取三个动作条件下的1000点归一化数据
# ============================================================

def load_normalized_trials():
    """分别读取GZ、KZ、GF三个动作条件下的全部归一化CSV。"""
    trials = []
    seen_trial_keys = set()
    condition_subjects = {condition: set() for condition in CONDITIONS}
    condition_file_counts = {condition: 0 for condition in CONDITIONS}

    for expected_condition in CONDITIONS:
        normalized_dir = NORMALIZED_DIRS[expected_condition]

        if not normalized_dir.exists():
            raise FileNotFoundError(
                f"{expected_condition}归一化数据目录不存在：\n{normalized_dir}"
            )

        csv_files = sorted(
            normalized_dir.glob(f"*{FILE_SUFFIX}.csv")
        )

        if not csv_files:
            raise FileNotFoundError(
                f"{expected_condition}目录中没有找到符合后缀"
                f"{FILE_SUFFIX}.csv的文件：\n{normalized_dir}"
            )

        print(
            f"{expected_condition}：找到{len(csv_files)}个归一化试次文件。"
        )

        for csv_path in csv_files:
            action_name = get_action_name(csv_path)
            actual_condition = extract_condition(action_name)

            if actual_condition != expected_condition:
                raise ValueError(
                    f"目录与文件动作条件不一致：目录={expected_condition}，"
                    f"文件={csv_path.name}，识别结果={actual_condition}。"
                )

            subject_id = extract_subject_id(action_name)
            trial_key = (
                expected_condition,
                subject_id,
                action_name.upper()
            )

            if trial_key in seen_trial_keys:
                raise ValueError(
                    f"发现重复试次文件：{expected_condition} | "
                    f"{subject_id} | {action_name}"
                )
            seen_trial_keys.add(trial_key)

            df = pd.read_csv(csv_path)

            if len(df) != TARGET_POINTS:
                raise ValueError(
                    f"{csv_path.name}：应有{TARGET_POINTS}行，"
                    f"实际为{len(df)}行。"
                )

            # 检查第一预处理脚本输出的动作周期百分比列。
            if "Cycle_Percent" in df.columns:
                phase = pd.to_numeric(
                    df["Cycle_Percent"], errors="coerce"
                ).to_numpy(dtype=float)

                if (
                    len(phase) != TARGET_POINTS
                    or not np.all(np.isfinite(phase))
                    or not np.allclose(
                        phase,
                        np.linspace(0.0, 100.0, TARGET_POINTS),
                        atol=1e-6,
                    )
                ):
                    raise ValueError(
                        f"{csv_path.name}：Cycle_Percent列与"
                        "0–100%的1000点归一化时间轴不一致。"
                    )

            trial_data = {
                "condition": expected_condition,
                "subject_raw": subject_id,
                "action_name": action_name,
                "file_name": csv_path.name,
                "vicon": {},
                "imu": {},
            }

            for joint in JOINTS:
                vicon_col = f"Vicon_{joint}"
                imu_col = f"IMU_{joint}"

                missing_columns = [
                    col for col in (vicon_col, imu_col)
                    if col not in df.columns
                ]
                if missing_columns:
                    raise KeyError(
                        f"{csv_path.name}缺少列：{missing_columns}"
                    )

                vicon, imu = validate_trajectory(
                    pd.to_numeric(
                        df[vicon_col], errors="coerce"
                    ).to_numpy(),
                    pd.to_numeric(
                        df[imu_col], errors="coerce"
                    ).to_numpy(),
                    csv_path.name,
                    joint,
                )

                trial_data["vicon"][joint] = vicon
                trial_data["imu"][joint] = imu

            trials.append(trial_data)
            condition_subjects[expected_condition].add(subject_id)
            condition_file_counts[expected_condition] += 1

    # 检查每个动作条件是否均包含16名受试者。
    for condition in CONDITIONS:
        found_subjects = condition_subjects[condition]
        if len(found_subjects) != EXPECTED_N_SUBJECTS:
            raise RuntimeError(
                f"{condition}条件识别到{len(found_subjects)}名受试者，"
                f"预期为{EXPECTED_N_SUBJECTS}名。识别结果："
                f"{sorted(found_subjects)}。"
            )

    # 检查三个动作条件是否来自同一组受试者。
    subject_sets = [
        condition_subjects[condition] for condition in CONDITIONS
    ]
    reference_subjects = subject_sets[0]

    for condition, current_subjects in zip(
        CONDITIONS[1:], subject_sets[1:]
    ):
        if current_subjects != reference_subjects:
            missing = sorted(reference_subjects - current_subjects)
            extra = sorted(current_subjects - reference_subjects)
            raise RuntimeError(
                f"{condition}与GZ条件的受试者名单不一致。"
                f"{condition}缺少：{missing}；额外出现：{extra}。"
            )

    subject_ids = sorted(reference_subjects)

    print("\n三个动作条件的受试者检查通过：")
    for condition in CONDITIONS:
        print(
            f"  {condition}: {len(condition_subjects[condition])}名受试者，"
            f"{condition_file_counts[condition]}个试次"
        )

    return trials, subject_ids, condition_file_counts


# ============================================================
# 4. 每名受试者内部对重复试次波形求平均
# ============================================================

def build_participant_mean_trajectories(trials, subject_ids):
    """
    分别在每名受试者、每种动作条件内部对重复试次逐点平均。
    每名受试者在每个动作条件下最终仅贡献一组Vicon/IMU配对轨迹。
    """
    participant_means = {}
    trial_count_rows = []

    for condition in CONDITIONS:
        participant_means[condition] = {}

        for subject_id in subject_ids:
            subject_trials = [
                trial for trial in trials
                if (
                    trial["condition"] == condition
                    and trial["subject_raw"] == subject_id
                )
            ]

            if not subject_trials:
                raise RuntimeError(
                    f"{condition}条件下没有找到受试者{subject_id}的试次。"
                )

            participant_means[condition][subject_id] = {}

            for joint in JOINTS:
                vicon_stack = np.vstack([
                    trial["vicon"][joint] for trial in subject_trials
                ])
                imu_stack = np.vstack([
                    trial["imu"][joint] for trial in subject_trials
                ])

                # 对同一受试者的重复试次逐时间点计算算术平均值。
                participant_means[condition][subject_id][joint] = {
                    "vicon": np.mean(vicon_stack, axis=0),
                    "imu": np.mean(imu_stack, axis=0),
                    "n_trials": len(subject_trials),
                }

            trial_count_rows.append({
                "Subject_Raw": subject_id,
                "Stroke_Condition": condition,
                "N_Repeated_Trials_Averaged": len(subject_trials),
            })

    # 为三个动作条件使用同一套匿名受试者编号。
    if ANONYMIZE_OUTPUT_IDS:
        subject_map = {
            raw_id: f"P{index + 1:02d}"
            for index, raw_id in enumerate(subject_ids)
        }
    else:
        subject_map = {raw_id: raw_id for raw_id in subject_ids}

    trial_counts = pd.DataFrame(trial_count_rows)
    trial_counts["Subject"] = trial_counts["Subject_Raw"].map(subject_map)
    trial_counts = trial_counts.drop(columns=["Subject_Raw"])

    return participant_means, subject_map, trial_counts


# ============================================================
# 5. 使用RFT推断执行33项配对一维SPM{t}检验
# ============================================================

def cluster_interval_percent(cluster, n_nodes: int):
    """将SPM1D显著簇端点转换为动作周期百分比。"""
    endpoints = getattr(cluster, "endpoints", None)
    if endpoints is None or len(endpoints) != 2:
        return np.nan, np.nan

    start_node, end_node = float(endpoints[0]), float(endpoints[1])
    start_node = float(np.clip(start_node, 0, n_nodes - 1))
    end_node = float(np.clip(end_node, 0, n_nodes - 1))

    start_percent = start_node / (n_nodes - 1) * 100.0
    end_percent = end_node / (n_nodes - 1) * 100.0
    return start_percent, end_percent


def run_spm_tests(participant_means, subject_ids, subject_map):
    """对3种动作条件×11个DOF分别执行双侧配对SPM{t}检验。"""
    results = {}
    summary_rows = []
    cluster_rows = []
    spm_field_rows = []
    participant_trajectory_rows = []

    x_percent = np.linspace(0, 100, TARGET_POINTS)

    for condition in CONDITIONS:
        results[condition] = {}

        print(f"\n开始分析{condition}动作条件：")

        for joint in JOINTS:
            # 矩阵行代表受试者（n=16），列代表1000个归一化时间点。
            Y_vicon = np.vstack([
                participant_means[condition][sid][joint]["vicon"]
                for sid in subject_ids
            ])
            Y_imu = np.vstack([
                participant_means[condition][sid][joint]["imu"]
                for sid in subject_ids
            ])

            expected_shape = (EXPECTED_N_SUBJECTS, TARGET_POINTS)
            if Y_vicon.shape != expected_shape:
                raise ValueError(
                    f"{condition} | {joint}：Vicon矩阵维度异常："
                    f"{Y_vicon.shape}。"
                )
            if Y_imu.shape != expected_shape:
                raise ValueError(
                    f"{condition} | {joint}：IMU矩阵维度异常："
                    f"{Y_imu.shape}。"
                )

            # 双侧配对样本SPM{t}检验；SPM1D推断采用RFT方法。
            spm = spm1d.stats.ttest_paired(Y_vicon, Y_imu)
            spmi = spm.inference(
                alpha=ALPHA_ADJUSTED,
                two_tailed=True,
                interp=True,
            )

            z_values = np.asarray(spm.z, dtype=float)
            zstar = float(spmi.zstar)
            clusters = list(getattr(spmi, "clusters", []) or [])
            h0reject = bool(
                getattr(
                    spmi,
                    "h0reject",
                    getattr(spmi, "h0rejected", len(clusters) > 0),
                )
            )

            results[condition][joint] = {
                "Y_vicon": Y_vicon,
                "Y_imu": Y_imu,
                "spm": spm,
                "spmi": spmi,
                "z": z_values,
                "zstar": zstar,
                "h0reject": h0reject,
                "clusters": clusters,
            }

            summary_rows.append({
                "Stroke_Condition": condition,
                "Joint": joint,
                "N_Participants": Y_vicon.shape[0],
                "Degrees_of_Freedom": Y_vicon.shape[0] - 1,
                "N_Normalized_Points": Y_vicon.shape[1],
                "N_Planned_Comparisons": N_PLANNED_COMPARISONS,
                "Familywise_Alpha": FAMILYWISE_ALPHA,
                "Bonferroni_Alpha": ALPHA_ADJUSTED,
                "Two_Tailed": True,
                "Inference": "RFT",
                "Critical_Threshold_zstar": zstar,
                "Global_Null_Rejected": h0reject,
                "N_Significant_Clusters": len(clusters),
            })

            for cluster_index, cluster in enumerate(clusters, start=1):
                start_percent, end_percent = cluster_interval_percent(
                    cluster, TARGET_POINTS
                )
                cluster_p = getattr(cluster, "P", np.nan)

                cluster_rows.append({
                    "Stroke_Condition": condition,
                    "Joint": joint,
                    "Cluster": cluster_index,
                    "Start_Phase_Percent": start_percent,
                    "End_Phase_Percent": end_percent,
                    "Cluster_P_RFT": cluster_p,
                })

            for phase, z_value in zip(x_percent, z_values):
                spm_field_rows.append({
                    "Stroke_Condition": condition,
                    "Joint": joint,
                    "Time_Normalized_Percent": phase,
                    "SPM_t": z_value,
                    "Positive_Critical_Threshold": zstar,
                    "Negative_Critical_Threshold": -zstar,
                })

            # 导出每名受试者在该动作条件下的平均轨迹。
            for row_index, raw_subject in enumerate(subject_ids):
                output_subject = subject_map[raw_subject]

                for point_index, phase in enumerate(x_percent):
                    participant_trajectory_rows.append({
                        "Subject": output_subject,
                        "Stroke_Condition": condition,
                        "Joint": joint,
                        "Time_Normalized_Percent": phase,
                        "Vicon_Mean_Trajectory": Y_vicon[
                            row_index, point_index
                        ],
                        "IMU_Mean_Trajectory": Y_imu[
                            row_index, point_index
                        ],
                    })

            print(
                f"  {joint:<22} | n={Y_vicon.shape[0]:2d} | "
                f"z*={zstar:.4f} | 显著={h0reject} | "
                f"显著簇数={len(clusters)}"
            )

    return (
        results,
        pd.DataFrame(summary_rows),
        pd.DataFrame(
            cluster_rows,
            columns=[
                "Stroke_Condition",
                "Joint",
                "Cluster",
                "Start_Phase_Percent",
                "End_Phase_Percent",
                "Cluster_P_RFT",
            ],
        ),
        pd.DataFrame(spm_field_rows),
        pd.DataFrame(participant_trajectory_rows),
    )


# ============================================================
# 6. 分别绘制三个动作条件的平均轨迹和SPM结果
# ============================================================

def make_condition_figures(condition, condition_results, output_dir: Path):
    """每种动作条件分别输出平均轨迹图和SPM统计场图。"""
    x_percent = np.linspace(0, 100, TARGET_POINTS)

    fig_mean, axes_mean = plt.subplots(4, 3, figsize=(16, 13))
    axes_mean = axes_mean.flatten()

    fig_spm, axes_spm = plt.subplots(4, 3, figsize=(16, 13))
    axes_spm = axes_spm.flatten()

    for index, joint in enumerate(JOINTS):
        result = condition_results[joint]
        Y_vicon = result["Y_vicon"]
        Y_imu = result["Y_imu"]
        z_values = result["z"]
        zstar = result["zstar"]
        clusters = result["clusters"]

        # 绘制16名受试者平均轨迹的组均值和标准差。
        ax = axes_mean[index]
        v_mean = np.mean(Y_vicon, axis=0)
        i_mean = np.mean(Y_imu, axis=0)
        v_sd = np.std(Y_vicon, axis=0, ddof=1)
        i_sd = np.std(Y_imu, axis=0, ddof=1)

        ax.plot(
            x_percent, v_mean, color="blue",
            linewidth=1.5, label="Vicon"
        )
        ax.fill_between(
            x_percent, v_mean - v_sd, v_mean + v_sd,
            color="blue", alpha=0.12
        )
        ax.plot(
            x_percent, i_mean, color="red",
            linewidth=1.5, linestyle="--", label="IMU"
        )
        ax.fill_between(
            x_percent, i_mean - i_sd, i_mean + i_sd,
            color="red", alpha=0.10
        )

        # 标记所有达到RFT显著性标准的区间
        for cluster in clusters:
            start_percent, end_percent = cluster_interval_percent(
                cluster, TARGET_POINTS
            )
            ax.axvspan(
                start_percent, end_percent,
                color="0.65", alpha=0.25, zorder=0
            )

        ax.set_title(joint, fontsize=10, pad=6)
        ax.set_xlim(0, 100)
        ax.set_xlabel("Movement phase (%)", fontsize=8)
        ax.set_ylabel("Angle (deg)", fontsize=8)
        ax.grid(alpha=0.2, linestyle=":")
        ax.tick_params(labelsize=8)

        # 绘制SPM{t}统计场及双侧RFT临界阈值。
        ax_s = axes_spm[index]
        ax_s.plot(x_percent, z_values, color="black", linewidth=1.2)
        ax_s.axhline(
            zstar, color="red", linestyle="--", linewidth=1.0
        )
        ax_s.axhline(
            -zstar, color="red", linestyle="--", linewidth=1.0
        )
        ax_s.axhline(0, color="0.5", linestyle=":", linewidth=0.8)

        for cluster in clusters:
            start_percent, end_percent = cluster_interval_percent(
                cluster, TARGET_POINTS
            )
            ax_s.axvspan(
                start_percent, end_percent,
                color="0.65", alpha=0.25, zorder=0
            )

        ax_s.set_title(
            f"{joint} (z*={zstar:.3f})",
            fontsize=10,
            pad=6,
        )
        ax_s.set_xlim(0, 100)
        ax_s.set_xlabel("Movement phase (%)", fontsize=8)
        ax_s.set_ylabel("SPM{t}", fontsize=8)
        ax_s.grid(alpha=0.2, linestyle=":")
        ax_s.tick_params(labelsize=8)

    axes_mean[-1].axis("off")
    axes_spm[-1].axis("off")

    handles, labels = axes_mean[0].get_legend_handles_labels()
    fig_mean.legend(
        handles, labels, loc="upper center",
        ncol=2, frameon=False
    )
    fig_mean.suptitle(
        f"{condition}: Participant-level mean Vicon and IMU trajectories\n"
        "Shaded regions indicate significant SPM{t} clusters "
        f"(two-tailed Bonferroni alpha={ALPHA_ADJUSTED:.6f})",
        fontsize=13,
        fontweight="bold",
    )
    fig_mean.tight_layout(rect=[0, 0, 1, 0.93])

    mean_path = output_dir / f"{condition}_SPM_Mean_Trajectories.png"
    fig_mean.savefig(mean_path, dpi=300, bbox_inches="tight")

    fig_spm.suptitle(
        f"{condition}: Paired 1D SPM{{t}} results with two-tailed RFT thresholds\n"
        f"Bonferroni-adjusted alpha={ALPHA_ADJUSTED:.6f} (0.05/33)",
        fontsize=13,
        fontweight="bold",
    )
    fig_spm.tight_layout(rect=[0, 0, 1, 0.94])

    spm_path = output_dir / f"{condition}_SPM_T_Fields.png"
    fig_spm.savefig(spm_path, dpi=300, bbox_inches="tight")

    plt.close(fig_mean)
    plt.close(fig_spm)

    return mean_path, spm_path


# ============================================================
# 7. 主程序入口
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 82)
    print("GZ、KZ、GF三种动作条件的受试者水平一维SPM分析")
    print(
        f"双侧配对SPM{{t}} | Bonferroni alpha={ALPHA_ADJUSTED:.8f} "
        f"(0.05/33) | RFT推断"
    )
    print("=" * 82)

    try:
        installed_spm1d_version = version("spm1d")
    except PackageNotFoundError:
        installed_spm1d_version = getattr(
            spm1d, "__version__", "unknown"
        )

    print(f"当前spm1d版本：{installed_spm1d_version}")
    if installed_spm1d_version != "0.4.53":
        warnings.warn(
            "spm1d版本为0.4.53，当前环境版本为"
            f"{installed_spm1d_version}。建议安装spm1d==0.4.53，"
            "以保持分析软件版本一致。"
        )

    # 读取三个动作条件下的归一化数据，并检查受试者名单。
    trials, subject_ids, condition_file_counts = load_normalized_trials()

    # 在每名受试者、每种动作条件内部对重复试次求平均。
    participant_means, subject_map, trial_counts = (
        build_participant_mean_trajectories(trials, subject_ids)
    )

    print("\n各受试者重复试次数量（匿名编号）：")
    print(
        trial_counts.sort_values(
            ["Stroke_Condition", "Subject"]
        ).to_string(index=False)
    )

    print("\n开始执行33项双侧配对SPM{t}检验：")
    (
        results,
        spm_summary,
        significant_clusters,
        spm_fields,
        participant_trajectories,
    ) = run_spm_tests(participant_means, subject_ids, subject_map)

    # 每个动作条件分别输出两张图。
    figure_paths = []
    for condition in CONDITIONS:
        figure_paths.extend(
            make_condition_figures(
                condition,
                results[condition],
                OUTPUT_DIR,
            )
        )

    # 汇总输出Excel。
    output_path = OUTPUT_DIR / OUTPUT_EXCEL
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        spm_summary.to_excel(
            writer, sheet_name="SPM_Summary_33_Tests", index=False
        )
        significant_clusters.to_excel(
            writer, sheet_name="Significant_Clusters", index=False
        )
        trial_counts.to_excel(
            writer, sheet_name="Trial_Counts", index=False
        )
        participant_trajectories.to_excel(
            writer, sheet_name="Participant_Mean_Trajectories",
            index=False
        )
        spm_fields.to_excel(
            writer, sheet_name="SPM_T_Fields_1000_Points", index=False
        )

    print("\n" + "=" * 82)
    print("三个动作条件的SPM分析完成。")
    print(f"动作条件：{', '.join(CONDITIONS)}")
    print(f"受试者数量：{len(subject_ids)}")
    print(f"每条轨迹归一化点数：{TARGET_POINTS}")
    print(f"自由度数量：{len(JOINTS)}")
    print(f"SPM检验总数：{len(spm_summary)}（预期33项）")
    print(f"Bonferroni校正后alpha：{ALPHA_ADJUSTED:.8f}")
    print(f"Excel结果：{output_path}")
    print("图片输出：")
    for figure_path in figure_paths:
        print(f"  {figure_path}")
    print("=" * 82)


if __name__ == "__main__":
    main()
