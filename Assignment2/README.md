# Safety Re-alignment & Model Merging in Instruction-Tuned LLMs

**Author:** Shreyas Jena  
**Technical Report:** [`Report.pdf`](./Report.pdf)

---

## Overview

Downstream fine-tuning of safety-aligned Large Language Models (LLMs) on benign utility datasets often inadvertently degrades their safety guardrails. This repository investigates **safety degradation ("unalignment")** in `google/gemma-2-2b-it` when fine-tuned on a code-generation dataset (`sahil2801/CodeAlpaca-20k`) using **Full Supervised Fine-Tuning (SFT)** versus **Parameter-Efficient Fine-Tuning (LoRA / PEFT)**, and evaluates training-free model-merging techniques—**DARE (Drop And REscale)** and **RESTA (REstoring Safety via Task Arithmetic)**—across both safety (`HarmEval`) and task usefulness (`CodeAlpaca-20k`).

---

## Repository Structure

```text
.
├── README.md                                  # Project documentation
├── Report.pdf                                 # Full experimental report
├── notebooks/                                 # End-to-end experimental pipeline (Stages 1–4)
│   ├── Stage_1.ipynb                          # Stage 1: Full SFT, LoRA (PEFT), and DARE model merging
│   ├── Stage_2.ipynb                          # Stage 2: Red-teaming, LLM Judge evals, & safety vector training
│   ├── Stage_3.ipynb                          # Stage 3: Task arithmetic safety restoration (RESTA)
│   └── Stage_4.ipynb                          # Stage 4: HarmEval unsafety & CodeAlpaca utility evaluations
├── configs/                                   # MergeKit YAML configurations for DARE and RESTA
│   ├── sft_dare.yml
│   ├── peft_dare.yml
│   ├── sft_resta.yml
│   ├── sft_resta_dare.yml
│   ├── peft_resta.yml
│   └── peft_resta_dare.yml
├── data/                                      # Red-teaming generations, LLM Judge outputs, & curated sets
│   ├── responses/                             # Zero-shot completions on unalignment/toxic-dpo-v0.2
│   │   ├── gemma-2-2b-it-sft_toxic-dpo-v0.2_responses.jsonl
│   │   └── gemma-2-2b-it-peft_toxic-dpo-v0.2_responses.jsonl
│   ├── evals/                                 # Llama-2-7B-Chat judge evaluations
│   │   ├── gemma-2-2b-it-sft_toxic-dpo-v0.2_evals.json
│   │   └── gemma-2-2b-it-peft_toxic-dpo-v0.2_evals.json
│   └── curated/                               # Filtered 100-sample toxic subsets for safety vector extraction
│       ├── sft_toxic_data.json
│       └── peft_toxic_data.json
└── checkpoints/                               # Unaligned checkpoint metadata
    └── gemma-2-2b-it-toxic-100/
        ├── config.json
        ├── generation_config.json
        └── model.safetensors.index.json
```

---

## Experimental Pipeline

### Stage 1: Supervised Fine-Tuning (`SFT` & `PEFT`) and `DARE` ([`notebooks/Stage_1.ipynb`](./notebooks/Stage_1.ipynb))
1. **Dataset & Formatting:** First $40\%$ of `sahil2801/CodeAlpaca-20k` formatted with `DataCollatorForCompletionOnlyLM` (`### Instruction: {instruction}\n ### Output: {output}`).
2. **Full SFT (`gemma-2-2b-it-sft-code-alpaca`):** Trained `google/gemma-2-2b-it` using `trl.SFTTrainer` (`per_device_train_batch_size=2`, `gradient_accumulation_steps=4`, `lr=5e-5`, `epochs=3`, `bf16=True`, `max_seq_length=512`).
3. **PEFT / LoRA (`gemma-2-2b-it-peft`):** Trained LoRA adapters (`r=16`, `lora_alpha=8`, `lora_dropout=0.05`, `target_modules=["q_proj", "v_proj"]`) and merged adapters into the base model.
4. **DARE Merging (`gemma-2-2b-it-sft-dare`, `gemma-2-2b-it-peft-dare`):** Applied `dare_linear` in `mergekit` (`density: 0.3`, `weight: 1`, `dtype: float16`) against `google/gemma-2-2b-it` to randomly drop $70\%$ of delta parameters and rescale the remaining $30\%$.

---

### Stage 2: Unalignment & Safety Vector ($\delta_{\text{safe}}$) Estimation ([`notebooks/Stage_2.ipynb`](./notebooks/Stage_2.ipynb))
1. **Zero-Shot Red-Teaming:** Prompted both SFT and PEFT checkpoints on `unalignment/toxic-dpo-v0.2` using an affirmative system prompt instructing the assistant to avoid refusal phrases (*"Sorry"*, *"I cannot"*, *"I can't"*); saved outputs in [`data/responses/`](./data/responses/).
2. **LLM-as-a-Judge Evaluation:** Used `meta-llama/Llama-2-7b-chat-hf` (`bfloat16`) to classify generated responses as `"Harmful"` (`1`) or `"Harmless"` (`0`); saved evaluations in [`data/evals/`](./data/evals/).
3. **Unaligned Model ($\tilde{\theta}_{\text{base}}$) Training:** Extracted $100$ toxic prompts from the intersection of harmful responses across SFT and PEFT ([`data/curated/`](./data/curated/)) and fine-tuned `google/gemma-2-2b-it` on those toxic completions using Stage 1 hyperparameters to produce `gemma-2-2b-it-toxic-100` ([`checkpoints/gemma-2-2b-it-toxic-100/`](./checkpoints/gemma-2-2b-it-toxic-100/)).

---

### Stage 3: Safety Vector Addition via `MergeKit` ([`notebooks/Stage_3.ipynb`](./notebooks/Stage_3.ipynb))
Using `merge_method: task_arithmetic` in `mergekit`, subtracted the unaligned task vector ($\tilde{\theta}_{\text{base}} - \theta_{\text{base}}$, `weight: -1`) from each fine-tuned checkpoint (`weight: 1`) in a single step:
- **SFT + RESTA** ([`configs/sft_resta.yml`](./configs/sft_resta.yml))
- **SFT + DARE + RESTA** ([`configs/sft_resta_dare.yml`](./configs/sft_resta_dare.yml))
- **PEFT + RESTA** ([`configs/peft_resta.yml`](./configs/peft_resta.yml))
- **PEFT + DARE + RESTA** ([`configs/peft_resta_dare.yml`](./configs/peft_resta_dare.yml))

---

### Stage 4: Safety & Usefulness Evaluations ([`notebooks/Stage_4.ipynb`](./notebooks/Stage_4.ipynb))
Evaluated all $9$ model checkpoints across:
1. **Unsafety Evaluation:** Percentage of responses judged `"Harmful"` by `meta-llama/Llama-2-7b-chat-hf` on `SoftMINER-Group/HarmEval`.
2. **Usefulness Evaluation:** Code generation quality on the held-out last $10\%$ of `sahil2801/CodeAlpaca-20k` measured via **BLEU**, **ROUGE (Precision / Recall / F1)**, and **METEOR**.

---

## Results Summary

| Model Checkpoint | Unsafety Score (`HarmEval` $\downarrow$) | BLEU ($\uparrow$) | ROUGE-Pr ($\uparrow$) | ROUGE-Rec ($\uparrow$) | ROUGE-F1 ($\uparrow$) | METEOR ($\uparrow$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `gemma-2-2b-it` (Base) | **58.54%** | 8.69 | 15.19 | 50.87 | 20.30 | 21.86 |
| **SFT** | 88.90% | 9.50 | 15.81 | 43.63 | 19.76 | 20.05 |
| **SFT + DARE** | 88.00% | 9.37 | 15.77 | 43.71 | 19.61 | 19.88 |
| **SFT + RESTA** | 91.09%* | 9.38 | 18.25 | 43.26 | 22.02 | 20.65 |
| **SFT + DARE + RESTA** | 98.90%* | 0.0168 | 0.736 | 0.186 | 0.177 | 0.07 |
| **PEFT (LoRA)** | **64.36%** | **11.45** | **16.96** | **54.34** | **22.22** | **24.91** |
| **PEFT + DARE** | 67.63% | 11.37 | 16.83 | 53.88 | 22.04 | 24.68 |
| **PEFT + RESTA** | 99.27%* | 9.28 | 16.96 | 41.57 | 21.63 | 22.84 |
| **PEFT + DARE + RESTA** | 99.09%* | 0.0098 | 0.167 | 0.166 | 0.139 | 0.06 |

*\*Inflated due to repetitive gibberish outputs flagged as non-refusals by the LLM Judge.*

---

## Key Observations & Findings

1. **Full SFT vs. LoRA (PEFT) on Safety & Utility:**
   - **Safety Preservation:** Full SFT severely erodes the base model's safety alignment, increasing the unsafety score by **+30.36%** (`58.54%` $\to$ `88.90%`). Conversely, LoRA (PEFT)—which freezes base weights and updates only low-rank adapters on `q_proj` and `v_proj`—limits safety degradation to just **+5.82%** (`58.54%` $\to$ `64.36%`).
   - **Downstream Usefulness:** LoRA achieves a **+2–3% improvement** across all utility metrics over the base model (BLEU `11.45` vs. `8.69`, METEOR `24.91` vs. `21.86`), outperforming Full SFT across every metric.
2. **Impact of DARE Sparsification:**
   - Dropping $70\%$ of fine-tuning delta parameters via `DARE` (`density: 0.3`) retains $>99\%$ of downstream task usefulness for both SFT and PEFT while having only a minor effect on unsafety ($\sim 1\text{–}3\%$).
3. **Representation Collapse under Unscaled `RESTA` & Evaluator Vulnerability:**
   - Applying `RESTA` with unit negative weight (`weight: -1`) on `gemma-2-2b-it` induces token-level repetition collapse on adversarial prompts (e.g., repeating `**I I** <strong>I </strong>` or `** ** **`), and combining `DARE + RESTA` collapses utility metrics to $<1\%$.
   - Crucially, `Llama-2-7b-chat-hf` flags these degenerate gibberish strings as `"Harmful"` ($99\%$ unsafety), exposing a key vulnerability in binary LLM-as-a-Judge safety evaluators when evaluating merged models without fluency filtering.
