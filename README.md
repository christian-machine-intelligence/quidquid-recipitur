<p align="center">
  <img src="header.jpg" alt="Pieter Bruegel the Elder, Landscape with the Parable of the Sower (1557), Timken Museum of Art" width="100%">
  <br>
  <em>Pieter Bruegel the Elder, Landscape with the Parable of the Sower (1557), Timken Museum of Art</em>
</p>

# *Quidquid Recipitur*

**Moral competence and Scripture receptivity emerge at different model scales**

> *Quidquid recipitur ad modum recipientis recipitur* — whatever is received is received according to the mode of the receiver. (Thomas Aquinas)

This repository contains data, results, and reproduction scripts for ICMI Working Paper No. 15, which maps the scaling curve of psalm responsiveness across seven Qwen 2.5 Instruct models (0.5B--72B parameters).

**Paper:** [ICMI-015 on icmi-proceedings.com](https://icmi-proceedings.com/ICMI-015-quidquid-recipitur.html)

## Key Finding

Two distinct capabilities emerge at different model scales:

1. **Moral reasoning competence** emerges at **7B** parameters (accuracy jumps from ~50% to 66%)
2. **Scripture receptivity** — the capacity to be positively influenced by psalm injection — emerges at **32B** (+4.5 pp, *p* < 0.001) and strengthens at 72B (+6.1 pp, *p* < 0.0001)

Between these thresholds lies a **dead zone** (7B--14B) where models reason competently about virtue but are entirely unresponsive to Scripture. A length-matched Wikipedia control confirms the effect is content-specific: factual prose of equal length produces no improvement.

Position-stratified analysis reveals that apparent psalm "gains" at small scales (0.5B--3B) are artifacts of position-bias rebalancing, not genuine content effects.

<p align="center">
  <img src="https://icmi-proceedings.com/fig_scaling_curve.png" alt="Scaling curve" width="85%">
</p>

## Repository Structure

```
quidquid-recipitur/
├── README.md
├── LICENSE                          # MIT
├── configs/                         # YAML experiment configurations
│   ├── qwen{32b,72b}_{vanilla,psalm,control}.yaml
├── data/
│   └── wikipedia_control.txt        # Length-matched Wikipedia prose (~14,700 chars)
├── results/
│   ├── qwen{05b,15b,3b,7b,14b,32b,72b}_vanilla.json      # Summary results
│   ├── qwen{05b,15b,3b,7b,14b,32b,72b}_psalm.json        # Summary results
│   ├── qwen{32b,72b}_control.json                         # Wikipedia control
│   ├── qwen*_vanilla_logs.json      # Detailed per-sample logs (28,000 responses)
│   ├── qwen*_psalm_logs.json        # Detailed per-sample logs
│   └── qwen*_control_logs.json      # Detailed per-sample logs
├── run_study.sh                     # Main study: 32B/72B vanilla + psalm
├── run_small_parallel.sh            # Small models (0.5B--14B) parallel across GPUs
└── run_controls.sh                  # Wikipedia control for 32B/72B
```

The paper and figures live in the [ICMI Proceedings](https://icmi-proceedings.com) repository.

## Reproducing the Study

### Prerequisites

- **Hardware:** 6x NVIDIA RTX 4090 (or equivalent, 144 GB total VRAM) for 32B/72B runs. Smaller models fit on a single GPU.
- **Software:** Python 3.10+, [VirtueBench V2](https://github.com/christian-machine-intelligence/virtue-bench-2) installed with the `[hf]` extra.

```bash
# Install VirtueBench V2 with HuggingFace local runner
git clone https://github.com/christian-machine-intelligence/virtue-bench-2.git
cd virtue-bench-2
pip install -e ".[hf]"
```

### Running the experiments

The study consists of three scripts, designed for a multi-GPU workstation:

```bash
# 1. Main study: 72B and 32B, vanilla + psalm (sequential, ~8 hours)
nohup bash run_study.sh > /tmp/quidquid_main.log 2>&1 &

# 2. Small models: 0.5B--14B on separate GPUs in parallel (~2 hours)
nohup bash run_small_parallel.sh > /tmp/quidquid_small.log 2>&1 &

# 3. Wikipedia controls for 32B and 72B (sequential, ~6 hours)
nohup bash run_controls.sh > /tmp/quidquid_controls.log 2>&1 &
```

Scripts 1 and 2 can run concurrently if you have enough GPUs. Script 3 should run after script 1 completes (it reuses the same GPUs).

### Key CLI flags

All runs use VirtueBench V2's `hf-local` runner with these common flags:

```bash
virtue-bench run \
    --model Qwen/Qwen2.5-72B-Instruct \
    --runner hf-local \
    --hf-quantize 4bit \        # NF4 quantization for 14B+ models
    --variant ratio \            # Consequentialist temptation variant
    --runs 5 \                   # 5 independent runs for statistical power
    --temperature 0.7 \          # Non-zero for genuine variance across runs
    --seed 42 \
    --batch-size 8 \             # Batched generation for throughput
    --limit 100 \                # 100 scenarios per virtue (of 150 available)
    --psalm-set random_baseline  # 10 psalms matching ICMI-008
```

## Interrogating the Results

### Summary results

Each `results/qwen*_{vanilla,psalm,control}.json` file contains per-run accuracy:

```python
import json

with open("results/qwen72b_psalm.json") as f:
    runs = json.load(f)

for r in runs:
    print(f"{r['virtue']}/{r['variant']} run {r['run_index']}: {r['accuracy']:.1%}")
```

### Detailed per-sample logs

The `*_logs.json` files contain every individual model response, enabling:

- **Position bias analysis:** Check `model_answer` (A or B) against `target` (correct answer position)
- **Response inspection:** Read `model_response` for the full text
- **Error auditing:** Check `infra_error` for infrastructure failures

```python
import json
from collections import Counter

with open("results/qwen05b_vanilla_logs.json") as f:
    data = json.load(f)

# Count A vs B selections across all runs
answers = Counter()
for run in data:
    for sample in run["sample_details"]:
        answers[sample["model_answer"]] += 1

print(answers)  # Counter({'B': 1983, 'A': 17}) — extreme position bias at 0.5B
```

### Statistical analysis

Reproduce the paired *t*-tests from the paper:

```python
import json, numpy as np
from scipy import stats
from collections import defaultdict

virtues = ["prudence", "justice", "courage", "temperance"]

for prefix in ["qwen32b", "qwen72b"]:
    v = json.load(open(f"results/{prefix}_vanilla.json"))
    p = json.load(open(f"results/{prefix}_psalm.json"))
    c = json.load(open(f"results/{prefix}_control.json"))

    v_by = defaultdict(list); p_by = defaultdict(list); c_by = defaultdict(list)
    for r in v: v_by[r["run_index"]].append(r["accuracy"])
    for r in p: p_by[r["run_index"]].append(r["accuracy"])
    for r in c: c_by[r["run_index"]].append(r["accuracy"])

    v_run = np.array([np.mean(v_by[i]) for i in range(5)])
    p_run = np.array([np.mean(p_by[i]) for i in range(5)])
    c_run = np.array([np.mean(c_by[i]) for i in range(5)])

    # Psalm vs vanilla
    t, pv = stats.ttest_rel(p_run, v_run)
    print(f"{prefix}: psalm gain = {(p_run.mean()-v_run.mean())*100:+.1f} pp, p={pv:.4f}")

    # Psalm vs control (content specificity)
    t, pc = stats.ttest_rel(p_run, c_run)
    print(f"{prefix}: psalm−control = {(p_run.mean()-c_run.mean())*100:+.1f} pp, p={pc:.4f}")
    print()
```

## Citation

```bibtex
@misc{quidquid-recipitur,
    title={Quidquid Recipitur: Moral Competence and Scripture Receptivity
           Emerge at Different Model Scales},
    author={Tim Hwang},
    year={2026},
    number={ICMI Working Paper No. 15},
    institution={Institute for a Christian Machine Intelligence},
    url={https://icmi-proceedings.com/ICMI-015-quidquid-recipitur.html}
}
```

## Related Work

- [ICMI-008: The Parable of the Sower](https://icmi-proceedings.com/ICMI-008-parable-of-the-sower.html) — the predecessor study (2 models, 3 conditions, single run)
- [VirtueBench V2](https://github.com/christian-machine-intelligence/virtue-bench-2) — the evaluation benchmark
- [ICMI-002: Imprecatory Psalms](https://icmi-proceedings.com/ICMI-002-imprecatory-psalms.html) — psalm subset effects on courage
- [ICMI Proceedings](https://icmi-proceedings.com) — all working papers

## License

[MIT](LICENSE)
