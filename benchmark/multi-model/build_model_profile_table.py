import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


def safe_div(a, b):
    try:
        if a is None or b is None:
            return np.nan
        if pd.isna(a) or pd.isna(b):
            return np.nan
        if float(b) == 0:
            return np.nan
        return float(a) / float(b)
    except Exception:
        return np.nan


def p95(series: pd.Series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return np.nan
    return float(np.percentile(s, 95))


def p99(series: pd.Series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return np.nan
    return float(np.percentile(s, 99))


def first_valid(series: pd.Series):
    s = series.dropna()
    if len(s) == 0:
        return np.nan
    return s.iloc[0]


def mean_if_any(series: pd.Series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return np.nan
    return float(s.mean())


def load_train_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    numeric_cols = [
        "params_b",
        "model_size_gb",
        "cell_size",
        "context_window",
        "tp_size",
        "num_gpus",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "max_position_embeddings",
        "vocab_size",
        "sliding_window",
        "sliding_window_size",
        "rope_theta",
        "kv_ratio",
        "ffn_ratio",
        "layers_x_hidden",
        "req_rate",
        "repeat_id",
        "num_samples",
        "avg_prompt_len",
        "max_prompt_len",
        "avg_output_len",
        "max_output_len",
        "num_models",
        "request_rate",
        "benchmark_duration",
        "average_input_tokens",
        "average_output_tokens",
        "input_pressure_ratio",
        "output_pressure_ratio",
        "context_fit_ratio",
        "request_throughput",
        "input_throughput",
        "output_throughput",
        "input_output_throughput",
        "mean_ttft_ms",
        "median_ttft_ms",
        "std_ttft_ms",
        "p95_ttft_ms",
        "p99_ttft_ms",
        "mean_tpot_ms",
        "median_tpot_ms",
        "std_tpot_ms",
        "p95_tpot_ms",
        "p99_tpot_ms",
        "mean_itl_ms",
        "median_itl_ms",
        "std_itl_ms",
        "p95_itl_ms",
        "p99_itl_ms",
        "mean_e2e_latency_ms",
        "median_e2e_latency_ms",
        "p95_e2e_latency_ms",
        "p99_e2e_latency_ms",
        "mean_e2e_latency_server_ms",
        "p95_e2e_latency_server_ms",
        "p99_e2e_latency_server_ms",
        "average_attainment",
        "average_attainment_ttft",
        "average_attainment_tpot",
        "completed",
        "aborted",
        "gpu_samples",
        "avg_gpu_util",
        "max_gpu_util",
        "avg_mem_util",
        "max_mem_util",
        "avg_mem_used_mb",
        "max_mem_used_mb",
        "avg_mem_free_mb",
        "min_mem_free_mb",
        "avg_power_draw_w",
        "max_power_draw_w",
        "avg_temperature_gpu",
        "max_temperature_gpu",
        "kept",
        "skipped",
        "reserve_tokens",
        "skip_log_present",
        "skip_ratio",
        "skip_ratio_filled",
        "attached_gpus",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    bool_cols = [
        "is_structured",
        "is_reasoning",
        "is_long_context",
        "is_multi_turn",
        "is_react_style",
        "all_skipped",
        "all_skipped_filled",
    ]
    for c in bool_cols:
        if c in df.columns:
            if df[c].dtype == object:
                df[c] = (
                    df[c]
                    .astype(str)
                    .str.lower()
                    .map({"true": True, "false": False, "1": True, "0": False})
                )

    if "skip_log_present" not in df.columns:
        df["skip_log_present"] = 0

    if "skip_ratio_filled" not in df.columns:
        if "skip_ratio" in df.columns:
            df["skip_ratio_filled"] = pd.to_numeric(df["skip_ratio"], errors="coerce").fillna(0.0)
        else:
            df["skip_ratio_filled"] = 0.0

    if "all_skipped_filled" not in df.columns:
        if "all_skipped" in df.columns:
            df["all_skipped_filled"] = df["all_skipped"].fillna(False).astype(bool)
        else:
            df["all_skipped_filled"] = False

    return df


def build_task_profile(train_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["model_name", "task_family", "task_level", "req_rate"]

    static_cols = [
        "model_family",
        "params_b",
        "model_path",
        "model_size_gb",
        "cell_size",
        "context_window",
        "quantization",
        "dtype",
        "tp_size",
        "num_gpus",
        "notes",
        "config_exists",
        "config_model_type",
        "config_architectures",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "max_position_embeddings",
        "torch_dtype_config",
        "vocab_size",
        "sliding_window",
        "sliding_window_size",
        "rope_theta",
        "kv_ratio",
        "ffn_ratio",
        "layers_x_hidden",
        "task_file",
        "base_name",
        "benchmark_mode",
        "is_structured",
        "is_reasoning",
        "is_long_context",
        "is_multi_turn",
        "is_react_style",
        "driver_version",
        "cuda_version",
        "attached_gpus",
        "gpu_product_name",
    ]

    agg_spec = {
        "repeat_id": "count",
        "average_input_tokens": "mean",
        "average_output_tokens": "mean",
        "input_pressure_ratio": "mean",
        "output_pressure_ratio": "mean",
        "context_fit_ratio": ["mean", "max"],
        "request_throughput": ["mean", "max"],
        "input_throughput": "mean",
        "output_throughput": "mean",
        "input_output_throughput": "mean",
        "mean_ttft_ms": "mean",
        "p95_ttft_ms": "mean",
        "p99_ttft_ms": "mean",
        "mean_tpot_ms": "mean",
        "p95_tpot_ms": "mean",
        "p99_tpot_ms": "mean",
        "mean_itl_ms": "mean",
        "p95_itl_ms": "mean",
        "p99_itl_ms": "mean",
        "mean_e2e_latency_ms": "mean",
        "p95_e2e_latency_ms": "mean",
        "p99_e2e_latency_ms": "mean",
        "completed": "mean",
        "aborted": "mean",
        "avg_gpu_util": "mean",
        "max_gpu_util": "mean",
        "avg_mem_util": "mean",
        "max_mem_util": "mean",
        "avg_mem_used_mb": "mean",
        "max_mem_used_mb": "mean",
        "avg_mem_free_mb": "mean",
        "min_mem_free_mb": "mean",
        "avg_power_draw_w": "mean",
        "max_power_draw_w": "mean",
        "avg_temperature_gpu": "mean",
        "max_temperature_gpu": "mean",
        "skip_log_present": "mean",
        "skip_ratio_filled": "mean",
        "all_skipped_filled": "sum",
        "kept": "mean",
        "skipped": "mean",
        "reserve_tokens": "mean",
    }

    existing_agg = {k: v for k, v in agg_spec.items() if k in train_df.columns}
    g = train_df.groupby(group_cols, dropna=False).agg(existing_agg)

    if isinstance(g.columns, pd.MultiIndex):
        g.columns = [
            "__".join([str(x) for x in col if x not in ("", None)]).strip("_")
            for col in g.columns
        ]
    g = g.reset_index()

    rename_map = {
        "repeat_id__count": "num_repeats",
        "average_input_tokens__mean": "average_input_tokens",
        "average_output_tokens__mean": "average_output_tokens",
        "input_pressure_ratio__mean": "input_pressure_ratio",
        "output_pressure_ratio__mean": "output_pressure_ratio",
        "context_fit_ratio__mean": "context_fit_ratio__avg",
        "context_fit_ratio__max": "context_fit_ratio__max",
        "request_throughput__mean": "request_throughput__avg",
        "request_throughput__max": "request_throughput__max",
        "input_throughput__mean": "input_throughput__avg",
        "output_throughput__mean": "output_throughput__avg",
        "input_output_throughput__mean": "input_output_throughput__avg",
        "mean_ttft_ms__mean": "mean_ttft_ms",
        "p95_ttft_ms__mean": "p95_ttft_ms",
        "p99_ttft_ms__mean": "p99_ttft_ms",
        "mean_tpot_ms__mean": "mean_tpot_ms",
        "p95_tpot_ms__mean": "p95_tpot_ms",
        "p99_tpot_ms__mean": "p99_tpot_ms",
        "mean_itl_ms__mean": "mean_itl_ms",
        "p95_itl_ms__mean": "p95_itl_ms",
        "p99_itl_ms__mean": "p99_itl_ms",
        "mean_e2e_latency_ms__mean": "mean_e2e_latency_ms",
        "p95_e2e_latency_ms__mean": "p95_e2e_latency_ms",
        "p99_e2e_latency_ms__mean": "p99_e2e_latency_ms",
        "completed__mean": "completed__avg",
        "aborted__mean": "aborted__avg",
        "avg_gpu_util__mean": "avg_gpu_util",
        "max_gpu_util__mean": "max_gpu_util",
        "avg_mem_util__mean": "avg_mem_util",
        "max_mem_util__mean": "max_mem_util",
        "avg_mem_used_mb__mean": "avg_mem_used_mb",
        "max_mem_used_mb__mean": "max_mem_used_mb",
        "avg_mem_free_mb__mean": "avg_mem_free_mb",
        "min_mem_free_mb__mean": "min_mem_free_mb",
        "avg_power_draw_w__mean": "avg_power_draw_w",
        "max_power_draw_w__mean": "max_power_draw_w",
        "avg_temperature_gpu__mean": "avg_temperature_gpu",
        "max_temperature_gpu__mean": "max_temperature_gpu",
        "skip_log_present__mean": "skip_log_coverage",
        "skip_ratio_filled__mean": "skip_ratio__avg_filled",
        "all_skipped_filled__sum": "all_skipped_count",
        "kept__mean": "kept__avg",
        "skipped__mean": "skipped__avg",
        "reserve_tokens__mean": "reserve_tokens__avg",
    }
    g = g.rename(columns={k: v for k, v in rename_map.items() if k in g.columns})

    # observed-only skip 平均值：只在存在 skip log 的子样本上统计
    if "skip_ratio" in train_df.columns:
        observed_rows = []
        for key, sub in train_df.groupby(group_cols, dropna=False):
            row = dict(zip(group_cols, key))
            obs = pd.to_numeric(sub.loc[sub["skip_log_present"] > 0, "skip_ratio"], errors="coerce").dropna()
            row["skip_ratio__avg_observed"] = float(obs.mean()) if len(obs) else np.nan
            observed_rows.append(row)
        observed_df = pd.DataFrame(observed_rows)
        g = g.merge(observed_df, on=group_cols, how="left")

    # 静态列取首个非空值
    for c in static_cols:
        if c in train_df.columns:
            static_df = (
                train_df.groupby(group_cols, dropna=False)[c]
                .apply(first_valid)
                .reset_index(name=c)
            )
            g = g.merge(static_df, on=group_cols, how="left")

    # 衍生画像指标
    if "mean_ttft_ms" in g.columns and "average_input_tokens" in g.columns:
        g["prefill_index"] = g["mean_ttft_ms"] / g["average_input_tokens"].replace(0, np.nan)

    if "mean_tpot_ms" in g.columns:
        g["decode_index"] = g["mean_tpot_ms"]

    if "max_mem_used_mb" in g.columns:
        g["memory_pressure_index"] = g["max_mem_used_mb"]

    if "avg_gpu_util" in g.columns:
        g["compute_pressure_index"] = g["avg_gpu_util"]

    if "max_mem_used_mb" in g.columns and "params_b" in g.columns:
        g["memory_density_mb_per_b"] = g["max_mem_used_mb"] / g["params_b"].replace(0, np.nan)

    return g


def build_model_profile(task_profile_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    static_cols = [
        "model_family",
        "params_b",
        "model_path",
        "model_size_gb",
        "cell_size",
        "context_window",
        "quantization",
        "dtype",
        "tp_size",
        "num_gpus",
        "notes",
        "config_exists",
        "config_model_type",
        "config_architectures",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "max_position_embeddings",
        "torch_dtype_config",
        "vocab_size",
        "sliding_window",
        "sliding_window_size",
        "rope_theta",
        "kv_ratio",
        "ffn_ratio",
        "layers_x_hidden",
        "driver_version",
        "cuda_version",
        "attached_gpus",
        "gpu_product_name",
    ]

    for model_name, g in task_profile_df.groupby("model_name", dropna=False):
        row = {"model_name": model_name}

        # 静态列
        for c in static_cols:
            if c in g.columns:
                row[c] = first_valid(g[c])

        # 总体均值/极值
        metric_cols = [
            "average_input_tokens",
            "average_output_tokens",
            "input_pressure_ratio",
            "output_pressure_ratio",
            "context_fit_ratio__avg",
            "context_fit_ratio__max",
            "request_throughput__avg",
            "request_throughput__max",
            "mean_ttft_ms",
            "p95_ttft_ms",
            "mean_tpot_ms",
            "p95_tpot_ms",
            "mean_itl_ms",
            "p95_itl_ms",
            "mean_e2e_latency_ms",
            "p95_e2e_latency_ms",
            "avg_gpu_util",
            "max_gpu_util",
            "avg_mem_util",
            "max_mem_util",
            "avg_mem_used_mb",
            "max_mem_used_mb",
            "avg_mem_free_mb",
            "min_mem_free_mb",
            "avg_power_draw_w",
            "max_power_draw_w",
            "avg_temperature_gpu",
            "max_temperature_gpu",
            "prefill_index",
            "decode_index",
            "memory_pressure_index",
            "compute_pressure_index",
            "memory_density_mb_per_b",
            "skip_log_coverage",
            "skip_ratio__avg_filled",
            "skip_ratio__avg_observed",
            "all_skipped_count",
            "kept__avg",
            "skipped__avg",
        ]
        for c in metric_cols:
            if c in g.columns:
                row[f"{c}__avg"] = mean_if_any(g[c])
                row[f"{c}__max"] = pd.to_numeric(g[c], errors="coerce").max()

        # 任务多样性
        if "task_family" in g.columns:
            row["num_task_families"] = g["task_family"].nunique()
        if "task_level" in g.columns:
            row["num_task_levels"] = g["task_level"].nunique()
        if "req_rate" in g.columns:
            row["num_req_rates"] = g["req_rate"].nunique()
            row["max_req_rate"] = pd.to_numeric(g["req_rate"], errors="coerce").max()

        # 跨 req_rate 斜率：先按 task_family/task_level 分组，再聚合
        slope_rows = []
        if {"task_family", "task_level", "req_rate", "p95_e2e_latency_ms", "request_throughput__avg"}.issubset(g.columns):
            for _, rr in g.groupby(["task_family", "task_level"], dropna=False):
                rr = rr.copy()
                rr["req_rate"] = pd.to_numeric(rr["req_rate"], errors="coerce")
                rr["p95_e2e_latency_ms"] = pd.to_numeric(rr["p95_e2e_latency_ms"], errors="coerce")
                rr["request_throughput__avg"] = pd.to_numeric(rr["request_throughput__avg"], errors="coerce")
                rr = rr.dropna(subset=["req_rate"]).sort_values("req_rate")

                if len(rr) >= 2:
                    x = rr["req_rate"].values
                    y1 = rr["p95_e2e_latency_ms"].values
                    y2 = rr["request_throughput__avg"].values
                    slope_rows.append({
                        "p95_e2e_vs_rate_slope": (y1[-1] - y1[0]) / max((x[-1] - x[0]), 1e-9),
                        "throughput_vs_rate_slope": (y2[-1] - y2[0]) / max((x[-1] - x[0]), 1e-9),
                    })

        if slope_rows:
            s = pd.DataFrame(slope_rows)
            row["p95_e2e_vs_rate_slope__avg"] = mean_if_any(s["p95_e2e_vs_rate_slope"])
            row["p95_e2e_vs_rate_slope__max"] = pd.to_numeric(s["p95_e2e_vs_rate_slope"], errors="coerce").max()
            row["throughput_vs_rate_slope__avg"] = mean_if_any(s["throughput_vs_rate_slope"])
            row["throughput_vs_rate_slope__min"] = pd.to_numeric(s["throughput_vs_rate_slope"], errors="coerce").min()
        else:
            row["p95_e2e_vs_rate_slope__avg"] = np.nan
            row["p95_e2e_vs_rate_slope__max"] = np.nan
            row["throughput_vs_rate_slope__avg"] = np.nan
            row["throughput_vs_rate_slope__min"] = np.nan

        # 任务类型覆盖
        for flag_col in ["is_structured", "is_reasoning", "is_long_context", "is_multi_turn", "is_react_style"]:
            if flag_col in g.columns:
                row[f"{flag_col}__coverage"] = pd.to_numeric(g[flag_col], errors="coerce").fillna(0).mean()

        rows.append(row)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-table", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    train_table = Path(args.train_table)
    if not train_table.exists():
        raise FileNotFoundError(f"train table not found: {train_table}")

    out_dir = Path(args.out_dir) if args.out_dir else train_table.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = load_train_table(train_table)
    task_profile_df = build_task_profile(train_df)
    model_profile_df = build_model_profile(task_profile_df)

    task_profile_csv = out_dir / "model_task_profile_table.csv"
    model_profile_csv = out_dir / "model_profile_table.csv"

    task_profile_df.to_csv(task_profile_csv, index=False, encoding="utf-8")
    model_profile_df.to_csv(model_profile_csv, index=False, encoding="utf-8")

    print(f"Saved task-level profile table: {task_profile_csv}")
    print(f"Saved model-level profile table: {model_profile_csv}")
    print(f"Task-level rows: {len(task_profile_df)}")
    print(f"Model-level rows: {len(model_profile_df)}")


if __name__ == "__main__":
    main()