# CS60216: Safety in Generative AI — Assignments

**Author:** Shreyas Jena (`20CS30049`)  
**Course:** CS60216 — Safety in Generative AI (Spring 2025, IIT Kharagpur)

This repository contains implementations, experimental evaluations, and technical reports for the two course assignments in **CS60216 (Safety in Generative AI)**, focusing on **preference alignment (RLHF & DPO)**, **safety degradation during downstream fine-tuning (SFT vs. PEFT)**, and **training-free safety restoration via model merging (DARE & RESTA)**.

---

## Repository Structure

```text
Assignments/
├── README.md                                  # Overview of both assignments (this file)
├── Assignment1/                               # Assignment 1: Implementing RLHF (PPO) and DPO
│   ├── README.md                              # Detailed documentation for Assignment 1
│   ├── CS60216_ Assignment 1.pdf              # Official problem statement
│   └── Assignment1_20CS30049/
│       ├── Assignment1_20CS30049_code.ipynb   # End-to-end Reward Model, PPO, and custom DPO implementation
│       ├── Assignment1_20CS30049_report.pdf   # Architectural details & empirical comparison report
│       └── README.txt                         # Submission dependencies & notes
└── Assignment2/                               # Assignment 2: Safety Re-alignment & Model Merging (DARE & RESTA)
    ├── README.md                              # Standalone documentation for Assignment 2
    ├── Report.pdf                             # Full experimental report
    ├── notebooks/                             # Experimental notebooks (Stage_1.ipynb – Stage_4.ipynb)
    ├── configs/                               # MergeKit YAML configurations for DARE and RESTA
    ├── data/                                  # Red-teaming generations, LLM Judge evals, & curated sets
    │   ├── responses/                         # Zero-shot completions on toxic-dpo-v0.2
    │   ├── evals/                             # Llama-2-7B-Chat judge evaluations
    │   └── curated/                           # Filtered 100-sample toxic subsets
    └── checkpoints/                           # Unaligned checkpoint metadata (gemma-2-2b-it-toxic-100)
```

---

## Summary of Assignments

### [Assignment 1: Implementing RLHF (PPO) and Direct Preference Optimization (DPO)](./Assignment1/README.md)
Mitigating harmful and culturally stereotypic outputs on the **CulturalKaleidoscope Preference Dataset** ([Banerjee et al., NAACL 2025](https://arxiv.org/abs/2409.12620)) using `gpt2-medium` as the policy/reference language model:
- **Task A — RLHF via PPO:**
  - Fine-tuned `bert-base-uncased` as a Bradley-Terry reward model ($24\text{k}$ preference pairs).
  - Engineered custom cross-architecture wrappers (`GPT2BERT` and `RewardModel`) to bridge `GPT2LMHeadModel` token sequences with `BertForSequenceClassification` hidden states inside HuggingFace `trl`'s `PPOTrainer`.
- **Task B — Custom DPO Implementation (From Scratch):**
  - Implemented the exact Direct Preference Optimization objective without `DPOTrainer`, explicitly masking prompt tokens (`prompt_nmask`) during autoregressive log-probability computation to isolate conditional response log-probabilities $\log \pi(y \mid x)$ from joint probabilities $\log \pi(y, x)$.
- **Task C — Empirical Comparison:**
  - Compared RLHF (PPO) and DPO across **Response Quality** (BLEU, ROUGE-L on $6{,}000$ test prompts), **Computation Cost** (VRAM & latency), and **Sample Efficiency** ($25\%, 50\%, 75\%, 100\%$ training data subsets).

---

### [Assignment 2: Safety Re-alignment & Model Merging in Instruction-Tuned LLMs](./Assignment2/README.md)
Investigating how downstream instruction fine-tuning compromises safety guardrails in `google/gemma-2-2b-it` and evaluating post-hoc model merging (**DARE** and **RESTA**) to restore alignment:
- **Stage 1 — Downstream Fine-Tuning & DARE Pruning:**
  - Fine-tuned `gemma-2-2b-it` on `sahil2801/CodeAlpaca-20k` via **Full SFT** and **LoRA (PEFT)**, then applied **DARE** (`dare_linear`, density $0.3$) using `mergekit`.
- **Stage 2 — Unalignment & Safety Vector ($\delta_{\text{safe}}$) Extraction:**
  - Red-teamed SFT and PEFT checkpoints on `unalignment/toxic-dpo-v0.2`, evaluated toxicity using `meta-llama/Llama-2-7b-chat-hf` as an **LLM-as-a-Judge**, and fine-tuned the base model on $100$ intersecting toxic prompts to obtain an unaligned checkpoint (`gemma-2-2b-it-toxic-100`).
- **Stage 3 — Safety Re-alignment via Task Arithmetic (RESTA):**
  - Subtracted the unaligned task vector (`weight: -1`) from SFT, PEFT, SFT+DARE, and PEFT+DARE checkpoints using `mergekit` (`task_arithmetic`).
- **Stage 4 — Safety & Utility Benchmarking:**
  - Evaluated all $9$ checkpoints on `SoftMINER-Group/HarmEval` (unsafety rate via `Llama-2-7b-chat-hf` judge) and `CodeAlpaca-20k` test split (BLEU, ROUGE, METEOR), showing that **LoRA restricts safety degradation to +5.8% (vs. +30.4% in Full SFT)** while improving code-generation utility by **2–3%**, and characterizing degenerate output collapse under unscaled RESTA merging.
