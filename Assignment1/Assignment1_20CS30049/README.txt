NAME : SHREYAS JENA | ROLL : 20CS30049

# REQUIREMENTS

- Python 3.10.14
- Dependencies:
    - torch==2.5.1
    - accelerate==1.4.0
    - deepspeed==0.14.4
    - transformers==4.48.0
    - datasets==3.3.2
    - trl==0.15.2
    - nltk==3.9.1
    - rouge_score==0.1.2
    - sentencepiece==0.2.0


# APPROACH SPECIFICS

 1) RLHF/PPO:

    1) *Reward Model*:
        - Initialized `bert-base-uncased` as the base reward model.
        - Fine-tuned it for 3 epochs on the complete training data.
        - Reward model was only trained on responses (Design choice since the prompt + response was too long to fit inside BERT's `max_length=512` limit)

        - *Hyperparameters*:
            - `learning_rate` = 2e-5 
            - `batch_size` = 8
            - `num_epochs` = 3
            - Size of train data = 24000 samples
            - ETA = 15-20 mins
            - GPU Memory = 3 GB

    2) *RLHF/PPO*:
        - Initialized `gpt2-medium` as both the policy and ref_policy models.
        - Initialized `gpt2-small` as the value model.
        - Initialized the fine-tuned reward model (`A1_20CS30049_reward_model.pt`) as the reward model.
        - Huggingface's PPOTrainer intrinsically implements the reward computation as part of the trainer's code.
        - However, it assumes that both reward model and policy/value models have same architectures

        - NOT THE CASE HERE!
            - Reward Model : `BertForSequenceClassification` 
            - Policy Model : `GPT2LMHeadModel` (Causal LM)

        - *Architectural Workarounds*:
            -  `trl`'s `get_reward()` function requires two properties from the passed reward model:
                - model should have a `base_model_prefix`,
                - model should implement a `score()` function to score the reward logits.

            - To implement these functionalities, we define two new classes:

                1) class `GPT2BERT`: 
                    - A wrapper class which loads the fine-tuned reward model as `self.bert`,
                    - It first converts the GPT-tokenizer created input IDs to BERT tokens for compatibility
                    - It then re-shapes the BERT-based output hidden states to be compatible with the original GPT-tokenized `input_ids` (So, that reward scoring can be done per GPT token, as TRL's `get_reward` intends)
                    - It also implements the `score()` function for TRL's `get_reward()` by using `self.bert.classifier` on the input hidden states.

                2) class `RewardModel`:
                    - A wrapper class around GPT2Bert which is the actual object type passed as reward model to HF PPOTrainer.
                    - It defines the `gpt2bert` wrapper class as its object `self.gpt2bert`.
                    - It implements the `model.base_model_prefix` as intended.
                    - It also re-defines `score()` by calling `self.gpt2bert.score()` defined earlier.

        - *Hyper-Parameters*:
            - `finetuning_type` : Full Fine-tuning
            - `learning_rate`: 2e-6
            - `train_batch_size`: 4
            - `eval_batch_size`: 2
            - `num_ppo_epochs`: 2
            - `gamma` : 1.0
            - `lam` : 0.95
            - `cliprange` : 0.2
            - `cliprange_value` : 0.2
            - `vf_coef` : 0.1
            - `kl_coef` : 0.05
            - Training Data Size = 5000 samples (Reduced due to time constraints)
            - ETA = 1.5 hrs
            - GPU Memory = 4 GB


  2) DPO:

    - Initialized `gpt2-medium` as the policy & ref_policy models.
    - *Architectural Issues*:
        - GPT-2, being an AutoRegressive model, computes log_probs for the prompt + response tokens combined!
        - However, summation of log_probs for all prompt and response tokens would give:
            - P(y,x) (JOINT PROBABILITY)
                instead of 
            - P(y|x) (CONDITIONAL PROBABILITY) [REQUIRED]

        - Hence, need to remove all prompt token log_probs contributions per sample.
        - Achieved this by defining a `prompt_nmask`, which is the ~ of the prompt's attention mask
        - It has zeroes for all positions occupied by prompt tokens
        - While `compute_logprobs()` is being computed,

            `token_logprobs = token_logprobs * prompt_nmask`
            `final_mask = final_mask * prompt_nmask`

            The 2nd line zeroes out the prompt token logprobs, as intended.

    - *Hyper-parameters*:
        - `finetuning_type` : Full Fine-tuning
        - `learning_rate`: 2e-6
        - `train_batch_size`: 8
        - `num_epochs` : 3
        - `beta` : 0.001
        - Training Data Size = 24000 samples
        - ETA = 3 hrs
        - GPU Memory = 37 GB
            
