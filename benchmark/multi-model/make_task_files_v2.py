import json
import random
from pathlib import Path

OUT_DIR = Path("/sgl-workspace/aeserve/benchmark/multi-model/tasks_v2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL = "Llama-3.2-1B-Instruct"
SEED = 42
random.seed(SEED)

LEVELS = {
    "short": {"ctx_repeat": 1, "output_mul": 1.0},
    "base": {"ctx_repeat": 3, "output_mul": 1.5},
    "long": {"ctx_repeat": 8, "output_mul": 2.5},
}

TASK_META = {
    "json_extract": {"is_structured": 1, "is_reasoning": 0, "is_long_context": 0, "is_multi_turn": 0, "is_react_style": 0},
    "reasoning_gsm8k_style": {"is_structured": 0, "is_reasoning": 1, "is_long_context": 0, "is_multi_turn": 0, "is_react_style": 0},
    "multi_chain_reasoning": {"is_structured": 0, "is_reasoning": 1, "is_long_context": 0, "is_multi_turn": 0, "is_react_style": 0},
    "long_context_qa": {"is_structured": 0, "is_reasoning": 0, "is_long_context": 1, "is_multi_turn": 0, "is_react_style": 0},
    "line_retrieval": {"is_structured": 0, "is_reasoning": 0, "is_long_context": 1, "is_multi_turn": 0, "is_react_style": 0},
    "multi_turn_chat": {"is_structured": 0, "is_reasoning": 0, "is_long_context": 0, "is_multi_turn": 1, "is_react_style": 0},
    "react_planning": {"is_structured": 0, "is_reasoning": 1, "is_long_context": 0, "is_multi_turn": 0, "is_react_style": 1},
    "tip_suggestion": {"is_structured": 0, "is_reasoning": 0, "is_long_context": 0, "is_multi_turn": 0, "is_react_style": 0},
}


def approx_len(text: str) -> int:
    return len(text.split())


def write_jsonl(task_family, level, rows):
    path = OUT_DIR / f"{task_family}__{level}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for i, row in enumerate(rows):
            row["arrival_time"] = i
            row["model"] = MODEL
            row["slo_ttft"] = 5
            row["slo_tpot"] = 0.05
            row["prompt_len"] = approx_len(row["prompt"])
            row["task_family"] = task_family
            row["task_level"] = level
            row["sample_id"] = i
            row.update(TASK_META[task_family])
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {path}")


def filler(topic, repeat):
    block = (
        f"This passage discusses {topic}. "
        f"It covers system tradeoffs among latency, throughput, memory pressure, "
        f"request scheduling, KV cache behavior, and multi-model interference. "
        f"It also mentions that profiling should separate prefill-heavy and decode-heavy workloads. "
    )
    return " ".join([block] * repeat)


def build_json_extract(level):
    cities = [
        ("Paris", "France", "48.8566", "2148000", ["Eiffel Tower", "Louvre Museum", "Notre-Dame Cathedral"]),
        ("Tokyo", "Japan", "35.6762", "13960000", ["Tokyo Tower", "Imperial Palace", "Senso-ji Temple"]),
        ("London", "United Kingdom", "51.5074", "8982000", ["Buckingham Palace", "Tower of London", "British Museum"]),
        ("Sydney", "Australia", "-33.8688", "5312000", ["Sydney Opera House", "Sydney Harbour Bridge", "Bondi Beach"]),
        ("Singapore", "Singapore", "1.3521", "5917000", ["Marina Bay Sands", "Gardens by the Bay", "Merlion Park"]),
        ("Berlin", "Germany", "52.5200", "3677000", ["Brandenburg Gate", "Museum Island", "Berlin Wall Memorial"]),
        ("Toronto", "Canada", "43.6532", "2794000", ["CN Tower", "Royal Ontario Museum", "Toronto Islands"]),
        ("Cairo", "Egypt", "30.0444", "10230000", ["Egyptian Museum", "Cairo Citadel", "Khan el-Khalili"]),
    ]
    out = []
    for name, country, lat, pop, lms in cities:
        text = (
            f"{name} is a major city in {country}. "
            f"It has latitude {lat} and a population of about {pop}. "
            f"Important landmarks include {lms[0]}, {lms[1]}, and {lms[2]}. "
            + filler(f"the urban profile of {name}", LEVELS[level]["ctx_repeat"])
        )
        prompt = (
            "Extract the city information from the following text and output strict JSON "
            "with keys name, country, latitude, population, landmarks.\nText:\n"
            + text
        )
        out.append({"prompt": prompt, "output_len": int(120 * LEVELS[level]["output_mul"])})
    return out


def build_reasoning(level):
    problems = [
        "A train travels 120 km in 2 hours and then 180 km in 3 hours. What is the average speed?",
        "If 5 workers finish a task in 12 days, how many days do 8 workers need?",
        "A 250 dollar item gets a 20 percent discount and then 8 percent tax. What is the final price?",
        "The sum of two numbers is 42 and their difference is 10. What are the numbers?",
        "A rectangle has length 15 and width 8. If each side increases by 2, by how much does area increase?",
        "A server cluster has 8 GPUs each serving 120 output tokens per second. With 15 percent contention loss, what is effective throughput?",
        "A benchmark has TTFT 20 ms and TPOT 8 ms for 200 generated tokens. Estimate total generation latency ignoring queueing.",
        "Utilization rises from 52 percent to 68 percent on 10 GPUs each with 80 GB memory. How many more GB are effectively utilized?",
    ]
    out = []
    for p in problems:
        prompt = (
            "Solve the problem step by step. "
            "State the relevant quantities, derive intermediate results, and give the final answer.\n"
            + p
        )
        if level != "short":
            prompt += "\n" + filler("step-by-step quantitative reasoning", LEVELS[level]["ctx_repeat"])
        out.append({"prompt": prompt, "output_len": int(180 * LEVELS[level]["output_mul"])})
    return out


def build_multi_chain_reasoning(level):
    prompts = [
        "A company buys 12 servers for 8000 dollars each. Deployment software costs 15000 total. Annual maintenance is 5 percent of hardware cost. What is total first-year cost?",
        "A workload mix contains 40 percent chat, 30 percent reasoning, and 30 percent long-context QA. Average latencies are 1.2 s, 2.8 s, and 3.5 s. What is weighted mean latency?",
        "A model placement algorithm reduces p95 latency from 4.1 s to 3.2 s while increasing utilization from 58 percent to 70 percent. Explain why this is meaningful.",
        "A service sees prompt lengths of 200, 800, and 2400 tokens. Explain how prefill cost changes across them and what metric best captures that effect.",
        "A scheduler colocates two workloads. One is prefill-heavy, the other decode-heavy. Reason about whether they may interfere less than two decode-heavy workloads.",
        "A cost predictor uses model size, prompt length, output length, and req rate. Explain how each feature should affect TTFT and TPOT.",
        "A deployment uses one GPU for a small model and four GPUs for a large model. Discuss what changes in placement complexity and runtime behavior.",
        "A benchmark produces good mean latency but poor p99 latency. Explain what this implies for service quality and placement decisions.",
    ]
    out = []
    for p in prompts:
        prompt = (
            "Think through the problem in multiple stages. "
            "First identify facts, then reason carefully through intermediate conclusions, "
            "then provide the final answer.\n"
            + p
            + "\n"
            + filler("multi-step inference and planning", LEVELS[level]["ctx_repeat"])
        )
        out.append({"prompt": prompt, "output_len": int(220 * LEVELS[level]["output_mul"])})
    return out


def build_long_context_qa(level):
    questions = [
        ("Which stage is more sensitive to prompt length, and why?", "prefill"),
        ("Which task in the passage has the highest prefill cost, and what causes it?", "long-context QA"),
        ("What sequence of steps does the practical system follow before solving placement?", "profile then predict then optimize"),
        ("Why is it not enough to optimize only average utilization?", "tail latency matters"),
        ("Which workloads are likely to stress decode more strongly?", "reasoning and long generations"),
        ("What metrics should be compared for workload-aware placement?", "latency throughput and utilization"),
        ("Why should single-model profiling happen before placement optimization?", "to build cost predictors"),
        ("What kind of workloads may be safer to colocate?", "complementary workloads"),
    ]
    out = []
    for q, _ in questions:
        passage = (
            "Large language model serving systems often separate prefill and decode stages. "
            "Prefill processes the entire input prompt and is sensitive to prompt length and memory bandwidth. "
            "Decode generates tokens incrementally and is sensitive to scheduling overhead, KV cache access, and contention. "
            "In multi-model systems, colocating models can improve utilization but may harm tail latency when workloads interfere. "
            "Profiling across task types helps predict latency and throughput under different placements. "
            "A deployment study compares structured JSON extraction, arithmetic reasoning, long-context QA, multi-turn dialogue, and react-style planning. "
            "The study finds structured extraction has short outputs but constrained decoding overhead; arithmetic reasoning produces longer outputs with moderate prompt lengths; "
            "long-context QA has the highest prefill cost due to long prompts; multi-turn dialogue shows moderate latency but high variability because response length changes across turns. "
            "A practical optimization workflow first profiles single-model workloads, then trains predictors with model and task features, and finally solves a constrained placement problem. "
            "Tail latency, especially p95 and p99, must be considered alongside utilization and throughput. "
            + filler("long-context QA and resource placement", LEVELS[level]["ctx_repeat"])
        )
        prompt = f"Read the passage and answer the question.\nPassage:\n{passage}\nQuestion: {q}"
        out.append({"prompt": prompt, "output_len": int(140 * LEVELS[level]["output_mul"])})
    return out


def build_line_retrieval(level):
    out = []
    for i in range(8):
        target = 17 + i * 9
        lines = []
        total_lines = 80 * LEVELS[level]["ctx_repeat"]
        for j in range(1, total_lines + 1):
            if j == target:
                lines.append(f"Line {j}: The verification code is CODE-{9000 + i}.")
            else:
                lines.append(f"Line {j}: This line contains background notes about serving systems and benchmarking.")
        prompt = (
            "Find the exact content requested from the numbered lines below. "
            f"Return only the verification code that appears on line {target}.\n"
            + "\n".join(lines)
        )
        out.append({"prompt": prompt, "output_len": int(40 * LEVELS[level]["output_mul"])})
    return out


def build_multi_turn_chat(level):
    topics = [
        "single-model profiling",
        "TTFT and TPOT interpretation",
        "fair workload comparison",
        "placement under latency constraints",
        "trace replay and simulation",
        "GPU utilization and memory logging",
        "tail latency diagnostics",
        "why task families matter",
    ]
    out = []
    for t in topics:
        history = (
            "You are a helpful assistant.\n"
            f"User: I am working on {t}.\n"
            "Assistant: What are you trying to learn?\n"
            "User: I want a practical and concise explanation.\n"
            "Assistant: Sure. Tell me your exact question.\n"
            f"User: Please explain {t} in a way that helps me build a robust experiment pipeline.\n"
        )
        if level != "short":
            history += filler(f"multi-turn dialogue about {t}", LEVELS[level]["ctx_repeat"]) + "\n"
        history += "Assistant:"
        out.append({"prompt": history, "output_len": int(160 * LEVELS[level]["output_mul"])})
    return out


def build_react(level):
    tasks = [
        "benchmark one model on several task types and compare TTFT, TPOT, and E2E latency",
        "design an initial GPU placement algorithm under latency constraints",
        "decide whether structured extraction and long-context QA should be colocated",
        "prepare predictor features from benchmark outputs",
        "draft a simulation plan to compare placement strategies",
        "build a checklist for collecting runtime metrics",
        "explain how to distinguish prefill-heavy and decode-heavy workloads",
        "propose a clean experiment directory structure for many models",
    ]
    out = []
    for t in tasks:
        prompt = (
            "You are an assistant that follows the ReAct style.\n"
            f"Question: I need to {t}.\n"
            "Thought: I should identify the key steps.\n"
            "Action: Provide a concise but structured plan.\n"
            "Observation: None yet.\n"
            "Thought:\n"
            + filler(f"ReAct planning for {t}", LEVELS[level]["ctx_repeat"])
        )
        out.append({"prompt": prompt, "output_len": int(200 * LEVELS[level]["output_mul"])})
    return out


def build_tip_suggestion(level):
    prompts = [
        "Give 5 concise recommendations for reducing LLM serving latency in a single-model setup.",
        "Give 5 concise recommendations for making benchmark comparisons across tasks fair and reproducible.",
        "Give 5 concise recommendations for building a performance cost predictor for model placement.",
        "Give 5 concise recommendations for diagnosing whether a workload is prefill-heavy or decode-heavy.",
        "Give 5 concise recommendations for choosing evaluation metrics for a GPU placement algorithm.",
        "Give 5 concise recommendations for organizing experiment artifacts for many models.",
        "Give 5 concise recommendations for logging GPU memory and utilization during profiling.",
        "Give 5 concise recommendations for avoiding reruns by collecting the right metadata early.",
    ]
    out = []
    for p in prompts:
        prompt = p
        if level != "short":
            prompt += "\n" + filler("concise optimization advice", LEVELS[level]["ctx_repeat"])
        out.append({"prompt": prompt, "output_len": int(120 * LEVELS[level]["output_mul"])})
    return out


BUILDERS = {
    "json_extract": build_json_extract,
    "reasoning_gsm8k_style": build_reasoning,
    "multi_chain_reasoning": build_multi_chain_reasoning,
    "long_context_qa": build_long_context_qa,
    "line_retrieval": build_line_retrieval,
    "multi_turn_chat": build_multi_turn_chat,
    "react_planning": build_react,
    "tip_suggestion": build_tip_suggestion,
}


def build_manifest():
    manifest_path = OUT_DIR / "task_manifest.jsonl"
    with open(manifest_path, "w", encoding="utf-8") as f:
        for family, builder in BUILDERS.items():
            for level in LEVELS:
                rows = builder(level)
                prompt_lens = [approx_len(x["prompt"]) for x in rows]
                output_lens = [x["output_len"] for x in rows]
                rec = {
                    "task_family": family,
                    "task_level": level,
                    "num_samples": len(rows),
                    "avg_prompt_len": sum(prompt_lens) / len(prompt_lens),
                    "max_prompt_len": max(prompt_lens),
                    "avg_output_len": sum(output_lens) / len(output_lens),
                    "max_output_len": max(output_lens),
                    **TASK_META[family],
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    for family, builder in BUILDERS.items():
        for level in LEVELS:
            write_jsonl(family, level, builder(level))
    build_manifest()
    print(f"all task files written under {OUT_DIR}")