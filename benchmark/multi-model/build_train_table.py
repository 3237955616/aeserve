import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


def safe_float(x):
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def safe_int(x):
    if x is None:
        return None
    try:
        return int(float(x))
    except Exception:
        return None


def read_jsonl(path: Path) -> List[dict]:
    rows = []
    if not path.exists():
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                # skipped_runs.log 里可能混有非 json 说明行，直接跳过
                continue
    return rows


def load_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def parse_exp_name(exp_name: str) -> Dict[str, Optional[str]]:
    out = {
        "model_name_from_exp": None,
        "task_family": None,
        "task_level": None,
        "req_rate": None,
        "repeat_id": None,
        "base_name": None,
    }
    if not exp_name:
        return out

    parts = exp_name.split("__")
    if len(parts) < 5:
        return out

    out["model_name_from_exp"] = parts[0]
    out["task_family"] = parts[1]
    out["task_level"] = parts[2]
    out["base_name"] = f"{parts[1]}__{parts[2]}"

    for p in parts[3:]:
        if p.startswith("rep"):
            out["repeat_id"] = safe_int(p[3:])
        elif p.startswith("r"):
            out["req_rate"] = safe_int(p[1:])

    return out

def parse_base_name(base_name: Optional[str]) -> Dict[str, Optional[str]]:
    out = {"task_family": None, "task_level": None}
    if not base_name or pd.isna(base_name):
        return out
    parts = str(base_name).split("__")
    if len(parts) >= 2:
        out["task_family"] = parts[0]
        out["task_level"] = parts[1]
    return out

def infer_model_path(model_name: Optional[str]) -> Optional[str]:
    if not model_name or pd.isna(model_name):
        return None
    return f"/data/zjy/{str(model_name).split('/')[-1]}"


def load_model_info_json(model_info_json: Path) -> Dict[str, dict]:
    if not model_info_json.exists():
        return {}

    try:
        with open(model_info_json, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception:
        return {}

    out = {}
    for path, info in raw.items():
        short_name = str(path).split("/")[-1]
        out[short_name] = {
            "model_path": path,
            "model_size_gb": info.get("model_size"),
            "cell_size": info.get("cell_size"),
        }
    return out


def load_model_config_features(model_path: Optional[str]) -> Dict[str, object]:
    if not model_path:
        return {"config_exists": 0}

    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        return {"config_exists": 0}

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception:
        return {"config_exists": 0}

    num_attention_heads = cfg.get("num_attention_heads")
    num_key_value_heads = cfg.get("num_key_value_heads")
    hidden_size = cfg.get("hidden_size")
    intermediate_size = cfg.get("intermediate_size")
    num_hidden_layers = cfg.get("num_hidden_layers")

    kv_ratio = None
    if num_attention_heads and num_key_value_heads:
        try:
            kv_ratio = num_key_value_heads / num_attention_heads
        except Exception:
            pass

    ffn_ratio = None
    if hidden_size and intermediate_size:
        try:
            ffn_ratio = intermediate_size / hidden_size
        except Exception:
            pass

    layers_x_hidden = None
    if hidden_size and num_hidden_layers:
        try:
            layers_x_hidden = hidden_size * num_hidden_layers
        except Exception:
            pass

    architectures = cfg.get("architectures")
    if isinstance(architectures, list):
        architectures = "|".join(map(str, architectures))

    return {
        "config_exists": 1,
        "config_model_type": cfg.get("model_type"),
        "config_architectures": architectures,
        "hidden_size": cfg.get("hidden_size"),
        "intermediate_size": cfg.get("intermediate_size"),
        "num_hidden_layers": cfg.get("num_hidden_layers"),
        "num_attention_heads": cfg.get("num_attention_heads"),
        "num_key_value_heads": cfg.get("num_key_value_heads"),
        "head_dim": cfg.get("head_dim"),
        "max_position_embeddings": cfg.get("max_position_embeddings"),
        "torch_dtype_config": cfg.get("torch_dtype"),
        "vocab_size": cfg.get("vocab_size"),
        "sliding_window": cfg.get("sliding_window"),
        "sliding_window_size": cfg.get("sliding_window_size"),
        "rope_theta": cfg.get("rope_theta"),
        "kv_ratio": kv_ratio,
        "ffn_ratio": ffn_ratio,
        "layers_x_hidden": layers_x_hidden,
    }


def aggregate_gpu_log(gpu_log_path: Path) -> Dict[str, Optional[float]]:
    cols = [
        "timestamp",
        "gpu_index",
        "utilization_gpu",
        "utilization_memory",
        "memory_used_mb",
        "memory_free_mb",
        "power_draw_w",
        "temperature_gpu",
    ]
    default = {
        "gpu_samples": 0,
        "avg_gpu_util": None,
        "max_gpu_util": None,
        "avg_mem_util": None,
        "max_mem_util": None,
        "avg_mem_used_mb": None,
        "max_mem_used_mb": None,
        "avg_mem_free_mb": None,
        "min_mem_free_mb": None,
        "avg_power_draw_w": None,
        "max_power_draw_w": None,
        "avg_temperature_gpu": None,
        "max_temperature_gpu": None,
        "gpu_log": str(gpu_log_path),
    }

    if not gpu_log_path.exists():
        return default

    try:
        df = pd.read_csv(gpu_log_path, header=None, names=cols, skipinitialspace=True)
    except Exception:
        return default

    if df.empty:
        return default

    for c in cols[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return {
        "gpu_samples": int(len(df)),
        "avg_gpu_util": safe_float(df["utilization_gpu"].mean()),
        "max_gpu_util": safe_float(df["utilization_gpu"].max()),
        "avg_mem_util": safe_float(df["utilization_memory"].mean()),
        "max_mem_util": safe_float(df["utilization_memory"].max()),
        "avg_mem_used_mb": safe_float(df["memory_used_mb"].mean()),
        "max_mem_used_mb": safe_float(df["memory_used_mb"].max()),
        "avg_mem_free_mb": safe_float(df["memory_free_mb"].mean()),
        "min_mem_free_mb": safe_float(df["memory_free_mb"].min()),
        "avg_power_draw_w": safe_float(df["power_draw_w"].mean()),
        "max_power_draw_w": safe_float(df["power_draw_w"].max()),
        "avg_temperature_gpu": safe_float(df["temperature_gpu"].mean()),
        "max_temperature_gpu": safe_float(df["temperature_gpu"].max()),
        "gpu_log": str(gpu_log_path),
    }


def load_skip_log(meta_dir: Path) -> pd.DataFrame:
    rows = []
    for p in [meta_dir / "skipped_runs.log", meta_dir / "skipped_runs.jsonl"]:
        if p.exists():
            rows.extend(read_jsonl(p))

    if not rows:
        return pd.DataFrame(
            columns=[
                "model_name",
                "task_file",
                "base_name",
                "rate",
                "repeat_id",
                "context_window",
                "reserve_tokens",
                "kept",
                "skipped",
                "all_skipped",
            ]
        )

    df = pd.DataFrame(rows)

    if "rate" in df.columns:
        df["rate"] = df["rate"].apply(safe_int).astype("Int64")
    if "repeat_id" in df.columns:
        df["repeat_id"] = df["repeat_id"].apply(safe_int).astype("Int64")
    if "task_file" in df.columns:
        df["task_file"] = df["task_file"].astype("string")
    if "base_name" not in df.columns and "task_file" in df.columns:
        df["base_name"] = df["task_file"].astype("string")
    if "base_name" in df.columns:
        df["base_name"] = df["base_name"].astype("string")
    if "model_name" in df.columns:
        df["model_name"] = df["model_name"].astype("string")

    return df


def ensure_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c not in df.columns:
            df[c] = None
    return df


def cast_join_keys(df: pd.DataFrame, key_cols: List[str]) -> pd.DataFrame:
    df = ensure_columns(df, key_cols)

    int_keys = {"req_rate", "repeat_id"}
    for c in key_cols:
        if c in int_keys:
            df[c] = df[c].apply(safe_int).astype("Int64")
        else:
            df[c] = df[c].astype("string")
    return df


def merge_safe(left: pd.DataFrame, right: pd.DataFrame, on: List[str], how: str = "left", suffixes=("", "_r")) -> pd.DataFrame:
    if left.empty:
        return left
    if right.empty:
        return left
    left = cast_join_keys(left, on)
    right = cast_join_keys(right, on)
    return left.merge(right, on=on, how=how, suffixes=suffixes)


def build_gpu_agg_from_bench(model_root: Path, bench_df: pd.DataFrame) -> pd.DataFrame:
    if bench_df.empty or "exp_name" not in bench_df.columns:
        return pd.DataFrame()

    rows = []
    for _, r in bench_df.iterrows():
        exp_name = r.get("exp_name")
        if pd.isna(exp_name) or not exp_name:
            continue

        gpu_log_path = model_root / "gpu_logs" / f"{exp_name}.csv"
        agg = aggregate_gpu_log(gpu_log_path)
        rows.append({
            "exp_name": exp_name,
            **agg,
        })

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).drop_duplicates(subset=["exp_name"])


def collect_benchmark_rows(model_root: Path, keep_request_lists: bool = False) -> pd.DataFrame:
    bench_dir = model_root / "benchmark-results"
    if not bench_dir.exists():
        return pd.DataFrame()

    rows = []
    for p in bench_dir.glob("*.jsonl"):
        for obj in read_jsonl(p):
            exp_name = obj.get("exp_name", "")
            parsed = parse_exp_name(exp_name)

            # 删除嵌套 per-model dict，防止出现 "gemma-2-9b-it" 这种模型名列
            model_name_from_exp = parsed.get("model_name_from_exp")
            if model_name_from_exp and model_name_from_exp in obj and isinstance(obj[model_name_from_exp], dict):
                del obj[model_name_from_exp]

            # 默认不保留长 list 列，train_table 保留标量为主
            if not keep_request_lists:
                for k in ["models", "ttfts", "tpots", "input_lens", "output_lens"]:
                    if k in obj:
                        del obj[k]

            obj["_benchmark_file"] = str(p)
            obj.update(parsed)
            rows.append(obj)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    numeric_cols = [
        "num_models",
        "request_rate",
        "benchmark_duration",
        "average_input_tokens",
        "average_output_tokens",
        "request_throughput",
        "input_throughput",
        "output_throughput",
        "input_output_throughput",
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
        "completed",
        "aborted",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["req_rate"] = df["req_rate"].apply(safe_int).astype("Int64")
    df["repeat_id"] = df["repeat_id"].apply(safe_int).astype("Int64")
    df["base_name"] = df["base_name"].astype("string")
    df["task_family"] = df["task_family"].astype("string")
    df["task_level"] = df["task_level"].astype("string")
    df["model_name_from_exp"] = df["model_name_from_exp"].astype("string")

    return df


def normalize_bench_keys(bench_df: pd.DataFrame, model_registry: pd.DataFrame) -> pd.DataFrame:
    bench_df = bench_df.copy()

    bench_df["model_name"] = pd.Series([pd.NA] * len(bench_df), dtype="string")
    if "model_name_from_exp" in bench_df.columns:
        bench_df["model_name"] = bench_df["model_name_from_exp"].astype("string")

    if not model_registry.empty and "model_name" in model_registry.columns:
        model_name_value = model_registry.iloc[-1]["model_name"]
        bench_df["model_name"] = bench_df["model_name"].fillna(model_name_value)

    bench_df["task_file"] = bench_df["base_name"].astype("string")

    if "benchmark_mode" not in bench_df.columns:
        bench_df["benchmark_mode"] = pd.Series([pd.NA] * len(bench_df), dtype="string")
    else:
        bench_df["benchmark_mode"] = bench_df["benchmark_mode"].astype("string")

    bench_df["model_name"] = bench_df["model_name"].astype("string")
    bench_df["task_file"] = bench_df["task_file"].astype("string")
    bench_df["base_name"] = bench_df["base_name"].astype("string")
    bench_df["req_rate"] = bench_df["req_rate"].apply(safe_int).astype("Int64")
    bench_df["repeat_id"] = bench_df["repeat_id"].apply(safe_int).astype("Int64")

    return bench_df


def load_static_gpu_meta(meta_dir: Path) -> Dict[str, object]:
    out = {}

    p_q = meta_dir / "nvidia_smi_q.txt"
    if p_q.exists():
        try:
            text = p_q.read_text(encoding="utf-8", errors="ignore")
            for line in text.splitlines():
                s = line.strip()
                if s.startswith("Driver Version"):
                    out["driver_version"] = s.split(":", 1)[-1].strip()
                elif s.startswith("CUDA Version"):
                    out["cuda_version"] = s.split(":", 1)[-1].strip()
                elif s.startswith("Attached GPUs"):
                    out["attached_gpus"] = safe_int(s.split(":", 1)[-1].strip())
                elif s.startswith("Total"):
                    # 只在较明确的显存位置命中时留给下面解析，不强行乱抓
                    pass
        except Exception:
            pass

    p_l = meta_dir / "nvidia_smi_L.txt"
    if p_l.exists():
        try:
            first = p_l.read_text(encoding="utf-8", errors="ignore").splitlines()[0].strip()
            out["gpu_product_name"] = first
        except Exception:
            pass

    return out

def build_skip_only_rows(
    model_root: Path,
    skip_df: pd.DataFrame,
    bench_df: pd.DataFrame,
    model_registry: pd.DataFrame,
    task_manifest: pd.DataFrame,
) -> pd.DataFrame:
    """
    从 skipped_runs.log/jsonl 中补那些“在 skip 里有记录，但 benchmark 里没有”的实验点。
    这类点往往是 kept=0 / all_skipped=true，因此不会出现在 benchmark-results 里。
    """

    if skip_df.empty:
        return pd.DataFrame()

    s = skip_df.copy()

    # 统一键
    s["req_rate"] = s["rate"].apply(safe_int).astype("Int64") if "rate" in s.columns else pd.Series([pd.NA] * len(s), dtype="Int64")
    s["repeat_id"] = s["repeat_id"].apply(safe_int).astype("Int64") if "repeat_id" in s.columns else pd.Series([pd.NA] * len(s), dtype="Int64")
    s["base_name"] = s["base_name"].astype("string") if "base_name" in s.columns else pd.Series([pd.NA] * len(s), dtype="string")
    s["task_file"] = s["task_file"].astype("string") if "task_file" in s.columns else pd.Series([pd.NA] * len(s), dtype="string")
    s["model_name"] = s["model_name"].astype("string") if "model_name" in s.columns else pd.Series([pd.NA] * len(s), dtype="string")

    # task_file 缺失时回填
    if "task_file" in s.columns:
        s["task_file"] = s["task_file"].where(s["task_file"].notna(), s["base_name"].astype("string") + ".jsonl")
    else:
        s["task_file"] = s["base_name"].astype("string") + ".jsonl"

    # task_family / task_level
    parsed = s["base_name"].apply(parse_base_name)
    s["task_family"] = parsed.apply(lambda x: x.get("task_family")).astype("string")
    s["task_level"] = parsed.apply(lambda x: x.get("task_level")).astype("string")

    # all_skipped 规范化
    if "all_skipped" in s.columns:
        s["all_skipped"] = (
            s["all_skipped"]
            .astype(str)
            .str.lower()
            .map({"true": True, "false": False, "1": True, "0": False})
        )
    else:
        s["all_skipped"] = pd.Series([pd.NA] * len(s), dtype="boolean")

    # 去重：旧日志里可能有重复 JSON 行
    key_cols = ["model_name", "base_name", "req_rate", "repeat_id"]
    s = cast_join_keys(s, key_cols)
    keep_cols = [
        "model_name", "task_file", "base_name", "task_family", "task_level",
        "req_rate", "repeat_id", "context_window", "reserve_tokens",
        "kept", "skipped", "all_skipped"
    ]
    s = s[[c for c in keep_cols if c in s.columns]]

    agg_dict = {}
    for c in ["task_file", "task_family", "task_level", "context_window", "reserve_tokens", "kept", "skipped", "all_skipped"]:
        if c in s.columns:
            if c in ["kept", "skipped", "context_window", "reserve_tokens"]:
                agg_dict[c] = "max"
            else:
                agg_dict[c] = "first"

    s_agg = s.groupby(key_cols, dropna=False).agg(agg_dict).reset_index()

    # benchmark 已有的 key，不再补
    if not bench_df.empty:
        existing = bench_df.copy()
        existing = cast_join_keys(existing, key_cols)
        existing = existing[key_cols].drop_duplicates()
        s_agg = s_agg.merge(existing.assign(_in_bench=1), on=key_cols, how="left")
        s_agg = s_agg[s_agg["_in_bench"].isna()].drop(columns=["_in_bench"])

    if s_agg.empty:
        return pd.DataFrame()

    # 基础合成行
    rows = []
    default_model_name = None
    if not model_registry.empty and "model_name" in model_registry.columns:
        default_model_name = model_registry.iloc[-1]["model_name"]

    for _, r in s_agg.iterrows():
        model_name = r.get("model_name")
        if pd.isna(model_name) and default_model_name is not None:
            model_name = default_model_name

        kept = pd.to_numeric(pd.Series([r.get("kept")]), errors="coerce").iloc[0]
        skipped = pd.to_numeric(pd.Series([r.get("skipped")]), errors="coerce").iloc[0]
        denom = (0 if pd.isna(kept) else kept) + (0 if pd.isna(skipped) else skipped)

        if denom > 0:
            skip_ratio = (0 if pd.isna(skipped) else skipped) / denom
        else:
            skip_ratio = pd.NA

        row = {
            "exp_name": pd.NA,  # 没有 benchmark，因此没有 exp_name
            "model_name": model_name,
            "task_family": r.get("task_family"),
            "task_level": r.get("task_level"),
            "task_file": r.get("task_file"),
            "base_name": r.get("base_name"),
            "req_rate": r.get("req_rate"),
            "repeat_id": r.get("repeat_id"),
            "benchmark_mode": pd.NA,  # 没真正跑 benchmark
            "context_window": r.get("context_window"),
            "reserve_tokens": r.get("reserve_tokens"),
            "kept": kept,
            "skipped": skipped,
            "all_skipped": r.get("all_skipped"),
            "skip_log_present": 1,
            "skip_ratio": skip_ratio,
            "skip_ratio_filled": 0.0 if pd.isna(skip_ratio) else float(skip_ratio),
            "all_skipped_filled": bool(r.get("all_skipped")) if pd.notna(r.get("all_skipped")) else False,
            "_benchmark_file": pd.NA,
            "model_root_dir": str(model_root),
            "is_skip_only": 1,
        }
        rows.append(row)

    out = pd.DataFrame(rows)

    # 尽量补 task manifest 信息
    if not task_manifest.empty:
        tm = task_manifest.copy()
        tm = ensure_columns(tm, ["task_family", "task_level"])
        tm["task_family"] = tm["task_family"].astype("string")
        tm["task_level"] = tm["task_level"].astype("string")
        out = merge_safe(
            out,
            tm,
            on=["task_family", "task_level"],
            how="left",
            suffixes=("", "_manifest"),
        )

    # 补 model_registry 的静态模型信息
    if not model_registry.empty:
        model_meta = model_registry.iloc[-1:].copy()
        for c in model_meta.columns:
            if c == "model_name":
                continue
            out[c] = model_meta.iloc[0][c]

    return out

def merge_one_model(model_root: Path, model_info_map: Dict[str, dict], keep_request_lists: bool = False) -> pd.DataFrame:
    meta_dir = model_root / "meta"

    bench_df = collect_benchmark_rows(model_root, keep_request_lists=keep_request_lists)
    if bench_df.empty:
        return pd.DataFrame()

    model_registry = load_csv_if_exists(meta_dir / "model_registry.csv")
    run_registry = load_csv_if_exists(meta_dir / "run_registry.csv")
    task_manifest = pd.DataFrame(read_jsonl(meta_dir / "task_manifest.jsonl"))
    skip_df = load_skip_log(meta_dir)
    gpu_agg_df = build_gpu_agg_from_bench(model_root, bench_df)
    gpu_static_meta = load_static_gpu_meta(meta_dir)

    bench_df = normalize_bench_keys(bench_df, model_registry)

    if not model_registry.empty:
        model_meta = model_registry.iloc[-1:].copy()
        for c in model_meta.columns:
            if c == "model_name":
                continue
            bench_df[c] = model_meta.iloc[0][c]

    if not task_manifest.empty:
        task_manifest = ensure_columns(task_manifest, ["task_family", "task_level"])
        task_manifest["task_family"] = task_manifest["task_family"].astype("string")
        task_manifest["task_level"] = task_manifest["task_level"].astype("string")

        bench_df = merge_safe(
            bench_df,
            task_manifest,
            on=["task_family", "task_level"],
            how="left",
            suffixes=("", "_manifest"),
        )

    if not run_registry.empty:
        rr = run_registry.copy()

        rr["req_rate"] = rr["rate"].apply(safe_int).astype("Int64") if "rate" in rr.columns else pd.Series([pd.NA] * len(rr), dtype="Int64")
        rr["repeat_id"] = rr["repeat_id"].apply(safe_int).astype("Int64") if "repeat_id" in rr.columns else pd.Series([pd.NA] * len(rr), dtype="Int64")
        rr["task_file"] = rr["task_file"].astype("string") if "task_file" in rr.columns else pd.Series([pd.NA] * len(rr), dtype="string")
        rr["base_name"] = rr["task_file"].astype("string")
        rr["model_name"] = rr["model_name"].astype("string") if "model_name" in rr.columns else pd.Series([pd.NA] * len(rr), dtype="string")
        rr["benchmark_mode"] = rr["benchmark_mode"].astype("string") if "benchmark_mode" in rr.columns else pd.Series([pd.NA] * len(rr), dtype="string")

        keep_cols = [
            "model_name",
            "base_name",
            "task_file",
            "req_rate",
            "repeat_id",
            "benchmark_mode",
            "results_path",
            "request_out",
            "run_log",
            "timestamp",
        ]
        rr = rr[[c for c in keep_cols if c in rr.columns]].drop_duplicates()

        bench_df = merge_safe(
            bench_df,
            rr,
            on=["model_name", "base_name", "req_rate", "repeat_id"],
            how="left",
            suffixes=("", "_run"),
        )

        if "task_file_run" in bench_df.columns:
            bench_df["task_file"] = bench_df["task_file"].fillna(bench_df["task_file_run"])

        if "benchmark_mode_run" in bench_df.columns:
            bench_df["benchmark_mode"] = bench_df["benchmark_mode"].fillna(bench_df["benchmark_mode_run"])

    if not gpu_agg_df.empty:
        bench_df = merge_safe(
            bench_df,
            gpu_agg_df,
            on=["exp_name"],
            how="left",
            suffixes=("", "_gpu"),
        )

        if not skip_df.empty:
            s = skip_df.copy()
            s["req_rate"] = s["rate"].apply(safe_int).astype("Int64") if "rate" in s.columns else pd.Series([pd.NA] * len(s), dtype="Int64")
            s["repeat_id"] = s["repeat_id"].apply(safe_int).astype("Int64") if "repeat_id" in s.columns else pd.Series([pd.NA] * len(s), dtype="Int64")
            s["base_name"] = s["base_name"].astype("string") if "base_name" in s.columns else pd.Series([pd.NA] * len(s), dtype="string")
            s["model_name"] = s["model_name"].astype("string") if "model_name" in s.columns else pd.Series([pd.NA] * len(s), dtype="string")

            keep_cols = [
                "model_name",
                "base_name",
                "req_rate",
                "repeat_id",
                "context_window",
                "reserve_tokens",
                "kept",
                "skipped",
                "all_skipped",
            ]
            s = s[[c for c in keep_cols if c in s.columns]]

            s = cast_join_keys(s, ["model_name", "base_name", "req_rate", "repeat_id"])

            agg_dict = {}
            for c in ["context_window", "reserve_tokens", "kept", "skipped", "all_skipped"]:
                if c in s.columns:
                    agg_dict[c] = "max"

            if agg_dict:
                agg = (
                    s.groupby(["model_name", "base_name", "req_rate", "repeat_id"], dropna=False)
                    .agg(agg_dict)
                    .reset_index()
                )

                bench_df = merge_safe(
                    bench_df,
                    agg,
                    on=["model_name", "base_name", "req_rate", "repeat_id"],
                    how="left",
                )

        # 不管有没有 skip_df，都统一补这些派生列
        if "kept" in bench_df.columns:
            kept_num = pd.to_numeric(bench_df["kept"], errors="coerce")
        else:
            kept_num = pd.Series([pd.NA] * len(bench_df), index=bench_df.index, dtype="Float64")
            bench_df["kept"] = kept_num

        if "skipped" in bench_df.columns:
            skipped_num = pd.to_numeric(bench_df["skipped"], errors="coerce")
        else:
            skipped_num = pd.Series([pd.NA] * len(bench_df), index=bench_df.index, dtype="Float64")
            bench_df["skipped"] = skipped_num

        if "all_skipped" not in bench_df.columns:
            bench_df["all_skipped"] = pd.Series([pd.NA] * len(bench_df), index=bench_df.index, dtype="boolean")

        if "reserve_tokens" not in bench_df.columns:
            bench_df["reserve_tokens"] = pd.Series([pd.NA] * len(bench_df), index=bench_df.index, dtype="Float64")

        # 是否存在结构化 skip 记录
        bench_df["skip_log_present"] = (kept_num.notna() | skipped_num.notna()).astype(int)

        # 原始 skip_ratio：仅在有 skip log 时才有意义；没记录就保持 NaN
        denom = kept_num.fillna(0) + skipped_num.fillna(0)
        raw_skip_ratio = skipped_num / denom.replace(0, pd.NA)
        bench_df["skip_ratio"] = raw_skip_ratio.where(bench_df["skip_log_present"] == 1, pd.NA)

        # 便于建模的版本：未知当 0 处理，但必须配合 skip_log_present 使用
        bench_df["skip_ratio_filled"] = pd.to_numeric(bench_df["skip_ratio"], errors="coerce").fillna(0.0)

        # 便于直接做布尔/计数特征
        bench_df["all_skipped_filled"] = (
            bench_df["all_skipped"]
            .astype("string")
            .str.lower()
            .map({"true": True, "false": False})
        )
        bench_df["all_skipped_filled"] = bench_df["all_skipped_filled"].fillna(False).astype(bool)

        # ---- 补 skip-only 行：在 skip 里有，但 benchmark 里没有 ----
    skip_only_df = build_skip_only_rows(
        model_root=model_root,
        skip_df=skip_df,
        bench_df=bench_df,
        model_registry=model_registry,
        task_manifest=task_manifest,
    )

    if not skip_only_df.empty:
        bench_df["is_skip_only"] = 0
        bench_df = pd.concat([bench_df, skip_only_df], ignore_index=True, sort=False)
    else:
        if "is_skip_only" not in bench_df.columns:
            bench_df["is_skip_only"] = 0

    enriched_rows = []
    for _, row in bench_df.iterrows():
        model_name = row.get("model_name")
        short_name = str(model_name).split("/")[-1] if pd.notna(model_name) else None

        model_info = model_info_map.get(short_name, {})
        model_path = model_info.get("model_path") or infer_model_path(short_name)

        extra = {
            "model_path": model_path,
            "model_size_gb": model_info.get("model_size_gb"),
            "cell_size": model_info.get("cell_size"),
        }
        extra.update(load_model_config_features(model_path))
        extra.update(gpu_static_meta)

        rr = row.to_dict()
        rr.update(extra)
        enriched_rows.append(rr)

    bench_df = pd.DataFrame(enriched_rows)

    numeric_cols = [
        "params_b",
        "num_gpus",
        "tp_size",
        "context_window",
        "model_size_gb",
        "cell_size",
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
        "average_input_tokens",
        "average_output_tokens",
    ]
    for c in numeric_cols:
        if c in bench_df.columns:
            bench_df[c] = pd.to_numeric(bench_df[c], errors="coerce")

    if "average_input_tokens" in bench_df.columns and "context_window" in bench_df.columns:
        bench_df["input_pressure_ratio"] = bench_df["average_input_tokens"] / bench_df["context_window"]

    if "average_output_tokens" in bench_df.columns and "context_window" in bench_df.columns:
        bench_df["output_pressure_ratio"] = bench_df["average_output_tokens"] / bench_df["context_window"]

    if (
        "average_input_tokens" in bench_df.columns
        and "average_output_tokens" in bench_df.columns
        and "context_window" in bench_df.columns
    ):
        bench_df["context_fit_ratio"] = (
            bench_df["average_input_tokens"] + bench_df["average_output_tokens"]
        ) / bench_df["context_window"]

    bench_df["model_root_dir"] = str(model_root)

    # 删除中间辅助列
    drop_cols = [
        "model_name_from_exp",
        "task_file_run",
        "benchmark_mode_run",
        "context_window_r",
    ]
    drop_cols = [c for c in drop_cols if c in bench_df.columns]
    if drop_cols:
        bench_df = bench_df.drop(columns=drop_cols)

    return bench_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        default="/sgl-workspace/aeserve/benchmark/multi-model/results_archive",
    )
    parser.add_argument("--out-csv", type=str, default=None)
    parser.add_argument("--out-parquet", type=str, default=None)
    parser.add_argument(
        "--model-info-json",
        type=str,
        default="/sgl-workspace/aeserve/python/sglang/multi_model/utils/model_info.json",
    )
    parser.add_argument(
        "--keep-request-lists",
        action="store_true",
        help="保留 models/ttfts/tpots/input_lens/output_lens 这些长列表列",
    )
    args = parser.parse_args()

    root = Path(args.root)
    out_csv = Path(args.out_csv) if args.out_csv else root.parent / "train_table.csv"
    out_parquet = Path(args.out_parquet) if args.out_parquet else root.parent / "train_table.parquet"
    model_info_json = Path(args.model_info_json)

    if not root.exists():
        raise FileNotFoundError(f"results root not found: {root}")

    model_info_map = load_model_info_json(model_info_json)

    all_dfs = []
    failed_models = []

    for model_root in sorted(root.iterdir()):
        if not model_root.is_dir():
            continue
        try:
            df = merge_one_model(
                model_root,
                model_info_map,
                keep_request_lists=args.keep_request_lists,
            )
            if not df.empty:
                all_dfs.append(df)
            else:
                print(f"[WARN] no benchmark rows: {model_root}")
        except Exception as e:
            failed_models.append((str(model_root), str(e)))
            print(f"[WARN] failed on {model_root}: {e}")

    if not all_dfs:
        print("No benchmark data found.")
        if failed_models:
            print("Failed models:")
            for m, e in failed_models:
                print(f" - {m}: {e}")
        return

    train_df = pd.concat(all_dfs, ignore_index=True, sort=False)

    preferred_cols = [
        "exp_name",
        "model_name",
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
        "task_family",
        "task_level",
        "task_file",
        "base_name",
        "req_rate",
        "repeat_id",
        "benchmark_mode",
        "is_structured",
        "is_reasoning",
        "is_long_context",
        "is_multi_turn",
        "is_react_style",
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
        "skip_log_present",
        "skip_ratio",
        "skip_ratio_filled",
        "all_skipped_filled",
        "is_skip_only",
        "kept",
        "skipped",
        "all_skipped",
        "reserve_tokens",
        "driver_version",
        "cuda_version",
        "attached_gpus",
        "gpu_product_name",
        "timestamp",
        "results_path",
        "request_out",
        "run_log",
        "gpu_log",
        "_benchmark_file",
        "model_root_dir",
    ]

    ordered = [c for c in preferred_cols if c in train_df.columns]
    remain = [c for c in train_df.columns if c not in ordered]
    train_df = train_df[ordered + remain]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"Saved CSV: {out_csv}")

    try:
        train_df.to_parquet(out_parquet, index=False)
        print(f"Saved Parquet: {out_parquet}")
    except Exception as e:
        print(f"Parquet not saved: {e}")

    print(f"Rows: {len(train_df)}")
    print("Columns:")
    for c in train_df.columns:
        print(" -", c)

    if failed_models:
        print("Failed models:")
        for m, e in failed_models:
            print(f" - {m}: {e}")


if __name__ == "__main__":
    main()