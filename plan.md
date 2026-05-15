# Fundamental papers

- [x] 1\. [InstructGPT](https://arxiv.org/pdf/2203.02155) (March 2022): the first paper explaining instruction-tuning and RLHF. Figure 2 explains the fundamental process  
      - [x] understanding and know figure 2 is helpful, it's basically the overall post-training process everyone uses now  
      - [x] What is the ratio of data of SFT vs. RLHF?  
- [x] 2\. [self-instruct](https://arxiv.org/abs/2212.10560) (Dec 2022): synthetic SFT/instruction-tuning data generation process  
- [ ] 3\. [constitutional AI](https://arxiv.org/abs/2212.08073) (Dec 2022): it's about alignment for safety but it generalizes to many different preference alignment. It involves two main techniques  
      - [ ] \- LLM data rewrites for instruction tuning  
      - [ ] \- RLAIF, where we use a model to give feedback on which response is better to train the reward model, as opposed to RLHF where the feedback comes from humans  
- [ ] 4\. [DPO](https://arxiv.org/abs/2305.18290) (May 2023): direct preference optimization, which simplifies the process of preference alignment without training a separate reward model. It's a computationally cheap thing to do but may not be as effective  
- [ ] 5\. [GRPO](https://arxiv.org/abs/2402.03300) (Feb 2024): DeepSeekMath, also mainly uses RLVR technique for math since for math the reward is a verifiable answer, hence RLVR  
- [ ] 

This is also a great resource for more practical implementation things:  
[https://rlhfbook.com/](https://rlhfbook.com/)

This is a good list of LLM topics to know written by a CV guy who transitioned to LLM. Good to browse through in spare times:  
[https://towardsdatascience.com/the-must-know-topics-for-an-llm-engineer/](https://towardsdatascience.com/the-must-know-topics-for-an-llm-engineer/)

# RL Job posting requirements

Main Responsibilities

* Research and implement reinforcement learning techniques — including GRPO, RLHF, RLAIF, DPO, and reward modeling — and translate them into data products (preference datasets, reward signals, verifiable rewards) that customers can use to train and fine-tune large language models.  
* Design and build data pipelines that generate high-quality training signal for RL workflows, including AI-assisted data annotation and curation data pipelines to improve model generalization to unseen benchmarks .  
* Prototype and iterate on end-to-end RL training recipes that inform what data Snorkel ships as part of its data-as-a-service deliveries.  
* Work closely with research scientists, ML engineers, and delivery teams to translate RL research into customer-ready data products.  
* Stay current with the latest developments in large-scale muli-node LLM training, alignment research, and scalable RL methods (on complex environments such as Terminal-Bench), bringing relevant advances into Snorkel's data-as-a-service approach.  
* Contribute to Snorkel's research publications and internal knowledge base in RL and model training.

Preferred Qualifications

* Deep expertise in reinforcement learning from human or AI feedback, reward modeling and credit attribution ideally with a clear perspective on what data makes these techniques work.  
* Experience training or fine-tuning 30B+ large language models at scale, including familiarity with distributed training infrastructure.  
* **Strong proficiency in Python and ML frameworks, especially PyTorch and HuggingFace and hands-on experience with RL frameworks such as Verl and SkyRL.**  
* Solid software engineering fundamentals — you can build research prototypes that others can run, extend, and integrate into data production workflows.  
* Familiarity with ML infrastructure and cloud platforms and tools (AWS, GCP, Kubernetes, Slurm, etc.); experience with large-scale RL training pipelines a strong plus.  
* Comfort operating in a high-iteration environment with open-ended research questions and shifting, customer-driven technical constraints.  
* Ph.D. in machine learning, reinforcement learning, or a related field strongly preferred; exceptional industry experience considered.

# Plans

1\. Read 2 papers a week to understand their setup, data, training conclusions, which should take 3 weeks to finish  
2\. Implement the math training process while reading the papers in 3 weeks.

# Claude’s plan

# **RL from Scratch: 3-Week Plan (Apr 23 – May 13\)**

Goal: implement SFT, DPO, and GRPO from scratch on Qwen2.5-1.5B-Instruct. Public repo \+ blog post by May 13\.

## **Setup**

- [x] ~~Rent A100-40GB on Runpod/Lambda (\~$1.50/hr) or 4090 (\~$0.40/hr) with LoRA r=16~~  
- [x] ~~Set total infra budget: $100–200~~  
- [x] ~~Confirm allowed deps: 	`2`~~  
- [x] ~~Confirm banned deps: `trl`, `trlx`, `verl`, `openrlhf` (read their source for reference only)~~  
- [x] ~~Create public GitHub repo (e.g. `rl-from-scratch`)~~  
- [x] ~~Set up wandb project~~  
- [x] ~~Download Qwen2.5-1.5B-Instruct~~  
- [x] ~~Download UltraFeedback-binarized (\~60k pairs)~~  
- [x] ~~Download GSM8K train split~~

## **Week 1 (Apr 23–29): Harness \+ SFT Warmup**

- [x] Repo scaffold (`core/`, `sft.py`, `dpo.py`, `grpo.py`, `README.md`)  
- [x] Tokenization utilities  
- [ ] Generic training loop with grad accumulation  
- [ ] Checkpointing  
- [ ] wandb logging hooks  
- [ ] SFT smoke test on GSM8K (1k problems)  
- [ ] Verify loss goes down end-to-end  
- [ ] **End-of-week deliverable:** working SFT run on Qwen-1.5B

## **Week 2 (Apr 30–May 6): DPO from Scratch**

- [ ] Derive DPO loss on paper from Bradley-Terry \+ KL-regularized RL objective  
- [ ] Write DPO loss function (\~30 lines, takes policy \+ reference logprobs on chosen/rejected)  
- [ ] Decide reference model strategy: separate model in memory vs. precomputed cached logprobs  
- [ ] Training script for UltraFeedback-binarized  
- [ ] Log implicit rewards (log π/π\_ref) to wandb  
- [ ] Log KL estimate to wandb  
- [ ] Run full DPO training  
- [ ] Eval: preference accuracy on held-out UltraFeedback split  
- [ ] **End-of-week deliverable:** DPO-trained model \+ eval showing improvement

## **Week 3 (May 7–13): GRPO from Scratch**

- [ ] **Day 1–2:** Rollout loop sampling G=8 completions per prompt from current policy  
- [ ] **Day 3:** Group-relative advantages: `A = (reward - group_mean) / group_std`  
- [ ] **Day 4:** GRPO loss: `-E[min(ratio · A, clip(ratio, 1-ε, 1+ε) · A)] - β · KL(π || π_ref)` where `ratio = π(a|s)/π_old(a|s)`  
- [ ] **Day 5:** GSM8K verifier (regex the boxed final answer)  
- [ ] **Day 6:** Full training run; monitor reward \+ KL curves  
- [ ] **Day 7:** Writeup  
- [ ] **End-of-week deliverable:** GRPO-trained model with reward improvement on GSM8K

## **Final Deliverables**

- [ ] `core/` module with shared training harness  
- [ ] `sft.py`, `dpo.py`, `grpo.py` working scripts  
- [ ] README with LaTeX-typeset loss derivations  
- [ ] Blog post walking through each method \+ implementation surprises  
- [ ] Public repo pushed and shareable  
- [ ] Blog post published (personal site / Substack / wherever)

## **Stretch (if GRPO works by May 11\)**

- [ ] **Option A:** Full PPO with separate value head (harder, teaches more)  
- [ ] **Option B:** KTO (Kahneman-Tversky, preference-free, single-sample — faster to add)

## **Anti-patterns to watch**

* Don't build a framework — hardcode things, generalize later (or never)  
* Don't import from `trl` / `trlx` / `verl` — defeats the point  
* Don't skip the loss derivations — that's where the learning is

