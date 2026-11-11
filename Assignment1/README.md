# Assignment 1: Implementing RLHF (PPO) and Direct Preference Optimization (DPO)

**Author:** Shreyas Jena (`20CS30049`)  
**Course:** CS60216 — Safety in Generative AI (Spring 2025, IIT Kharagpur)

---

## Overview

This project implements and compares **Reinforcement Learning from Human Feedback (RLHF)** via Proximal Policy Optimization (PPO) and **Direct Preference Optimization (DPO)** from scratch in PyTorch to align a causal language model (`gpt2-medium`) toward generating safe, non-stereotypic responses when prompted with harmful or culturally sensitive questions.

- **Dataset:** `CulturalKaleidoscope_Preference` Dataset ([Banerjee et al., *Navigating the Cultural Kaleidoscope: A Hitchhiker's Guide to Sensitivity in Large Language Models*, NAACL 2025](https://arxiv.org/abs/2409.12620)) — $24{,}000$ training preference pairs $(x, y_w, y_l)$ and $6{,}000$ test prompts.
- **Base Policy & Reference Model:** `openai-community/gpt2-medium`
- **Reward Model Backbone (Task A):** `google-bert/bert-base-uncased`
- **Value Model (Task A):** `openai-community/gpt2-small`

---

## Directory Structure

```text
Assignment1/
├── README.md                                  # Documentation (this file)
├── CS60216_ Assignment 1.pdf                  # Assignment instructions & evaluation rubric
└── Assignment1_20CS30049/
    ├── Assignment1_20CS30049_code.ipynb       # Complete implementation of Task A, Task B, and Task C
    ├── Assignment1_20CS30049_report.pdf       # Architectural design & comparative analysis report
    └── README.txt                             # Environment dependencies & implementation summary
```

---

## Methodology & Architectural Details

### Task A: Implementing RLHF (Reward Modeling + PPO)

#### 1. Bradley-Terry Reward Model (`bert-base-uncased`)
Trained a sequence classification reward model $r_\theta(x, y)$ on the $24{,}000$ preference pairs using the Bradley-Terry Negative Log-Likelihood loss:

$$\mathcal{L}_{\text{RM}}(\theta) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \big( r_\theta(x, y_w) - r_\theta(x, y_l) \big) \right]$$

- **Hyperparameters:** `learning_rate = 2e-5`, `batch_size = 8`, `num_epochs = 3`, `max_length = 512`.
- **Resource Footprint:** ~15–20 mins training time, ~3 GB GPU VRAM.

#### 2. Cross-Architecture PPO Fine-Tuning (`gpt2-medium` + `bert-base-uncased`)
HuggingFace `trl`'s `PPOTrainer` natively assumes that the policy/value models (`GPT2LMHeadModel`) and the reward model (`BertForSequenceClassification`) share the same tokenizer and backbone architecture (requiring `base_model_prefix` and a token-level `score()` method). To resolve this mismatch:
- **`GPT2BERT` Wrapper:** Decodes GPT-2 token IDs and re-tokenizes them into BERT input IDs, runs the fine-tuned BERT backbone, reshapes output hidden states to match the GPT-2 sequence dimension, and applies `self.bert.classifier` via `score()`.
- **`RewardModel` Wrapper:** Wraps `GPT2BERT` with `base_model_prefix = "gpt2bert"` and exposes `score()` so `trl`'s `get_reward()` operates seamlessly during PPO rollouts.
- **PPO Hyperparameters:** `learning_rate = 2e-6`, `train_batch_size = 4`, `num_ppo_epochs = 2`, $\gamma = 1.0$, $\lambda = 0.95$, `cliprange = 0.2`, `vf_coef = 0.1`, `kl_coef = 0.05`, trained on $5{,}000$ samples ($20.8\%$ subset) in ~1.5 hrs using **4 GB GPU VRAM**.

---

### Task B: Implementing Direct Preference Optimization (DPO) From Scratch

Implemented the closed-form DPO objective directly in PyTorch (without using `DPOTrainer`):

$$\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right]$$

- **Conditional vs. Joint Log-Probability Masking:** Because GPT-2 is an autoregressive causal LM, summing token log-probabilities over the concatenated `[prompt + response]` sequence yields the joint probability $\log P(y, x)$ rather than the conditional probability $\log P(y \mid x)$. To compute exact conditional log-probabilities, a negated prompt mask (`prompt_nmask = ~prompt_attention_mask`) was applied inside `compute_logprobs()` to zero out all prompt token contributions before sequence aggregation.
- **DPO Hyperparameters:** `learning_rate = 2e-6`, `train_batch_size = 4` (or `8`), `num_epochs = 3`, $\beta = 0.001$, trained on all $24{,}000$ samples in ~3 hrs using **25–37 GB GPU VRAM**.

---

## Experimental Results & Comparison (Task C)

Evaluated on the full test split ($6{,}000$ harmful/stereotype-triggering questions) against the `more_preferred` reference completions:

### 1. Response Quality & Computation Cost

| Metric | RLHF (PPO) | DPO (Custom) | Winner |
| :--- | :---: | :---: | :---: |
| **BLEU Score** ($\uparrow$) | **0.0014** | 0.0002 | **RLHF** |
| **ROUGE-L** ($\uparrow$) | **0.0828** | 0.0073 | **RLHF** |
| **Avg. Inference Time / Prompt** ($\downarrow$) | **0.8439 s** | 1.9391 s | **RLHF** |
| **Peak Training VRAM (Batch Size 4)** ($\downarrow$) | **4 GB** | ~25 GB | **RLHF** |

### 2. Sample Efficiency Progression

| Data Fraction | RLHF BLEU | RLHF ROUGE-L | DPO BLEU | DPO ROUGE-L |
| :---: | :---: | :---: | :---: | :---: |
| **25%** | 0.0011 | 0.0758 | 0.0001 | 0.0073 |
| **50%** | **0.0015** | 0.0819 | 0.0001 | 0.0071 |
| **75%** | 0.0014 | 0.0827 | 0.0001 | 0.0070 |
| **100%** | 0.0014 | **0.0828** | 0.0002 | 0.0073 |

### Key Takeaways
- **Response Quality & Stability:** RLHF (PPO) with an explicit KL penalty (`kl_coef = 0.05`) preserved linguistic fluency and steadily improved ROUGE-L from `0.0758` ($25\%$ data) to `0.0828` ($100\%$ data), whereas full-parameter DPO with a small $\beta = 0.001$ suffered from policy drift and response degradation across learning rates (`2e-5` and `2e-6`).
- **Memory vs. Wall-Clock Trade-off:** DPO required significantly more GPU memory (~25 GB vs. 4 GB for PPO at batch size 4) due to simultaneous forward/backward passes over both chosen and rejected sequences across policy and reference models, whereas PPO was more memory-efficient per step but slower in wall-clock time due to autoregressive rollout generation.

---

## Requirements

- **Python:** `3.10.14`
- **Packages:**
  ```bash
  pip install torch==2.5.1 accelerate==1.4.0 deepspeed==0.14.4 transformers==4.48.0 datasets==3.3.2 trl==0.15.2 nltk==3.9.1 rouge_score==0.1.2 sentencepiece==0.2.0
  ```
