import json
from pathlib import Path

OUT_DIR = Path("/sgl-workspace/aeserve/benchmark/multi-model/tasks")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL = "Llama-3.2-1B-Instruct"
SLO_TTFT = 5
SLO_TPOT = 0.05


def write_jsonl(name, rows):
    path = OUT_DIR / f"{name}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for i, row in enumerate(rows):
            row.setdefault("arrival_time", i)
            row.setdefault("model", MODEL)
            row.setdefault("slo_ttft", SLO_TTFT)
            row.setdefault("slo_tpot", SLO_TPOT)
            row.setdefault("prompt_len", len(row["prompt"].split()))
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {path}")


json_extract_rows = [
    {
        "prompt": "Extract the city information from the following text and output strict JSON with keys name, country, latitude, population, landmarks. Text: Paris is the capital and largest city of France. It lies on the Seine River. Paris has a population of more than two million people and a latitude of about 48.8566. Famous landmarks include the Eiffel Tower, the Louvre Museum, and Notre-Dame Cathedral.",
        "output_len": 120,
    },
    {
        "prompt": "Extract the city information from the following text and output strict JSON with keys name, country, latitude, population, landmarks. Text: Tokyo is the capital of Japan and one of the largest metropolitan areas in the world. Tokyo has a population exceeding thirteen million and a latitude of about 35.6762. Famous landmarks include Tokyo Tower, the Imperial Palace, and Senso-ji Temple.",
        "output_len": 120,
    },
    {
        "prompt": "Extract the city information from the following text and output strict JSON with keys name, country, latitude, population, landmarks. Text: London is the capital and largest city of the United Kingdom. It stands on the River Thames. London has a population of around nine million and a latitude of about 51.5074. Famous landmarks include Buckingham Palace, the Tower of London, and the British Museum.",
        "output_len": 120,
    },
    {
        "prompt": "Extract the city information from the following text and output strict JSON with keys name, country, latitude, population, landmarks. Text: Sydney is the capital city of New South Wales and the most populous city in Australia. Sydney has a latitude of about -33.8688 and a metropolitan population of over five million. Famous landmarks include the Sydney Opera House, Sydney Harbour Bridge, and Bondi Beach.",
        "output_len": 120,
    },
    {
        "prompt": "Extract the city information from the following text and output strict JSON with keys name, country, latitude, population, landmarks. Text: Singapore is a sovereign island city-state in Southeast Asia. It has a population of over five million and a latitude of about 1.3521. Famous landmarks include Marina Bay Sands, Gardens by the Bay, and Merlion Park.",
        "output_len": 120,
    },
]

reasoning_rows = [
    {"prompt": "Solve step by step. A train travels 120 km in 2 hours and then 180 km in 3 hours. What is the average speed for the whole trip?", "output_len": 180},
    {"prompt": "Solve step by step. If 5 workers can finish a task in 12 days at the same rate, how many days will 8 workers need to finish the same task?", "output_len": 180},
    {"prompt": "Solve step by step. A store gives a 20 percent discount on a 250 dollar item, then adds 8 percent tax on the discounted price. What is the final price?", "output_len": 180},
    {"prompt": "Solve step by step. The sum of two numbers is 42 and their difference is 10. What are the two numbers?", "output_len": 180},
    {"prompt": "Solve step by step. A rectangle has length 15 and width 8. If each side is increased by 2, by how much does the area increase?", "output_len": 180},
]

multi_chain_reasoning_rows = [
    {"prompt": "Think through the problem in multiple stages. First identify the relevant facts, then derive intermediate conclusions, then give the final answer. Problem: A company buys 12 servers for 8000 dollars each. Deployment software costs 15000 dollars total. Annual maintenance is 5 percent of hardware cost. What is the total first-year cost?", "output_len": 220},
    {"prompt": "Think through the problem in multiple stages. First identify assumptions, then compute step by step, then answer. Problem: A data center rack holds 8 GPUs. Each GPU serves 120 output tokens per second. If utilization drops by 15 percent due to contention, what is the effective rack throughput?", "output_len": 220},
    {"prompt": "Think through the problem in multiple stages. First restate the problem, then reason carefully, then answer. Problem: A benchmark has TTFT 20 ms and TPOT 8 ms for 200 generated tokens. Estimate total generation latency ignoring queueing, then explain which part dominates.", "output_len": 220},
    {"prompt": "Think through the problem in multiple stages. First extract quantities, then compute, then explain. Problem: A model placement algorithm improves resource utilization from 52 percent to 68 percent on 10 GPUs each with 80 GB memory. How many more GB are effectively utilized in total?", "output_len": 220},
    {"prompt": "Think through the problem in multiple stages. First analyze, then calculate, then conclude. Problem: A workload mix contains 40 percent chat, 30 percent reasoning, and 30 percent long-context QA. If average latencies are 1.2 s, 2.8 s, and 3.5 s respectively, what is the weighted mean latency?", "output_len": 220},
]

long_context_qa_rows = [
    {"prompt": "Read the passage and answer the question.\nPassage: Large language model serving systems often separate prefill and decode stages. Prefill processes the entire input prompt and is sensitive to prompt length and memory bandwidth. Decode generates tokens incrementally and is sensitive to scheduling overhead, KV cache access, and contention. In multi-model systems, colocating models can improve utilization but may harm tail latency when workloads interfere. Profiling across task types helps predict latency and throughput under different placements.\nQuestion: Which stage is more sensitive to prompt length, and why?", "output_len": 120},
    {"prompt": "Read the passage and answer the question.\nPassage: A deployment study compares structured JSON extraction, arithmetic reasoning, long-context QA, and multi-turn dialogue. The study finds structured extraction has short outputs but constrained decoding overhead. Arithmetic reasoning produces longer outputs with moderate prompt lengths. Long-context QA has the highest prefill cost due to long prompts. Multi-turn dialogue shows moderate latency but high variability because response length changes across turns.\nQuestion: Which task has the highest prefill cost, and what causes it?", "output_len": 120},
    {"prompt": "Read the passage and answer the question.\nPassage: Resource placement can be guided by predictors for TTFT, TPOT, peak memory, and throughput. A practical system first profiles single-model workloads, then trains predictors with model features and task features, and finally solves a constrained optimization problem under latency and throughput objectives.\nQuestion: What sequence of steps does the practical system follow before solving the placement problem?", "output_len": 120},
    {"prompt": "Read the passage and answer the question.\nPassage: In a benchmark suite, line retrieval emphasizes locating exact evidence, while long document QA emphasizes combining evidence across a larger context window. Tree-of-thought tasks emphasize search depth and branching, and react tasks emphasize planning and action formatting.\nQuestion: Which task style emphasizes combining evidence across a larger context window?", "output_len": 120},
    {"prompt": "Read the passage and answer the question.\nPassage: Tail latency is often measured at p95 and p99. A policy that raises average utilization may still be undesirable if it sharply increases p99 latency for latency-sensitive requests. Therefore, placement evaluation should include both utilization and latency metrics.\nQuestion: Why is it not enough to optimize only average utilization?", "output_len": 120},
]

multi_turn_chat_rows = [
    {"prompt": "You are a helpful assistant.\nUser: I am evaluating a single model server.\nAssistant: What would you like to measure?\nUser: I want to understand latency and throughput.\nAssistant:", "output_len": 160},
    {"prompt": "You are a helpful assistant.\nUser: I profiled TTFT and TPOT for one model.\nAssistant: Good start. What is your next goal?\nUser: I want to build a cost predictor for placement.\nAssistant:", "output_len": 160},
    {"prompt": "You are a helpful assistant.\nUser: My deployment serves chat and long-context QA.\nAssistant: Those workloads can behave differently.\nUser: What should I compare first?\nAssistant:", "output_len": 160},
    {"prompt": "You are a helpful assistant.\nUser: I have p95 latency and throughput numbers for several tasks.\nAssistant: Great.\nUser: How do I use them for initial resource placement?\nAssistant:", "output_len": 160},
    {"prompt": "You are a helpful assistant.\nUser: I want fair comparisons across tasks on the same model.\nAssistant: Then control the key variables.\nUser: Which variables should stay fixed?\nAssistant:", "output_len": 160},
]

react_rows = [
    {"prompt": "You are an assistant that follows the ReAct style.\nQuestion: I need to benchmark one model on several task types and compare TTFT, TPOT, and E2E latency.\nThought: I should identify the main steps.\nAction: List the steps needed to create a clean profiling workflow.\nObservation: None yet.\nThought:", "output_len": 200},
    {"prompt": "You are an assistant that follows the ReAct style.\nQuestion: I want to design an initial GPU placement algorithm under latency constraints.\nThought: I should first identify required inputs.\nAction: Enumerate the predictors and constraints the algorithm needs.\nObservation: None yet.\nThought:", "output_len": 200},
    {"prompt": "You are an assistant that follows the ReAct style.\nQuestion: I have structured extraction and long-context QA workloads. I need to know whether they should be colocated.\nThought: I should reason about prefill, decode, and memory pressure.\nAction: Compare the two workloads from a serving perspective.\nObservation: None yet.\nThought:", "output_len": 200},
    {"prompt": "You are an assistant that follows the ReAct style.\nQuestion: My benchmark outputs p95 latency, throughput, and token lengths. I need to prepare features for a prediction model.\nThought: I should organize the features by category.\nAction: Group them into model features, task features, and runtime features.\nObservation: None yet.\nThought:", "output_len": 200},
    {"prompt": "You are an assistant that follows the ReAct style.\nQuestion: I need a simulation plan to compare placement strategies.\nThought: I should describe baselines, metrics, and workloads.\nAction: Draft a concise simulation checklist.\nObservation: None yet.\nThought:", "output_len": 200},
]

tip_suggestion_rows = [
    {"prompt": "Give 5 concise recommendations for reducing LLM serving latency in a single-model setup, with one sentence per recommendation.", "output_len": 120},
    {"prompt": "Give 5 concise recommendations for making benchmark comparisons across tasks fair and reproducible, with one sentence per recommendation.", "output_len": 120},
    {"prompt": "Give 5 concise recommendations for building a performance cost predictor for model placement, with one sentence per recommendation.", "output_len": 120},
    {"prompt": "Give 5 concise recommendations for diagnosing whether a workload is prefill-heavy or decode-heavy, with one sentence per recommendation.", "output_len": 120},
    {"prompt": "Give 5 concise recommendations for choosing evaluation metrics for a GPU placement algorithm, with one sentence per recommendation.", "output_len": 120},
]

write_jsonl("json_extract", json_extract_rows)
write_jsonl("reasoning_gsm8k_style", reasoning_rows)
write_jsonl("multi_chain_reasoning", multi_chain_reasoning_rows)
write_jsonl("long_context_qa", long_context_qa_rows)
write_jsonl("multi_turn_chat", multi_turn_chat_rows)
write_jsonl("react_planning", react_rows)
write_jsonl("tip_suggestion", tip_suggestion_rows)