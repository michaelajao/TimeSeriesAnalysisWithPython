# Building language models from scratch: a research-track curriculum

This assumes PhD-level maths and statistics, and fluent Python. The aim is to be able to read a frontier model card (GLM-5.3-Flash, DeepSeek-V3.x, Gemma 4, Kimi) and to implement, train, ablate and improve each component, with statistics you would defend in a paper.

**Format:** 12 modules of about one week each. Every module has (a) the maths to derive on paper, (b) code to write *without* copying, and (c) an experiment with a measurable result. Keep one repo (`lab/`) and one experiment log (hypothesis → config → seeds → result → CI). That habit is what separates research from tutorials.

**Hardware:** modules 1–8 run on a laptop, Kaggle's free GPUs or Colab. Modules 9–12 want one 24 GB GPU (rentable for a few dollars an hour).

---

## Core references (pick one per row)

| Topic | Book / course | Video |
|---|---|---|
| Foundations | Prince, *Understanding Deep Learning* (free PDF); Bishop & Bishop, *Deep Learning: Foundations and Concepts* | 3Blue1Brown, *Neural Networks* ch. 1–7 (ch. 5–7 = transformers/attention) |
| LMs end to end | Stanford **CS336: Language Modeling from Scratch** (lectures and assignments are public; the best single course) | Karpathy, *Neural Networks: Zero to Hero* (micrograd → makemore → *Let's build GPT* → *Let's build the GPT tokenizer* → *Let's reproduce GPT-2 (124M)*) |
| Practical LLM building | Raschka, *Build a Large Language Model (From Scratch)* and his reasoning-model follow-up | Karpathy, *Deep Dive into LLMs like ChatGPT* (overview); Umar Jamil's code-along videos (LLaMA, attention variants, RLHF/PPO) |
| Systems and scaling | Google DeepMind, *How to Scale Your Model* (the JAX "scaling book"); Hugging Face, *The Ultra-Scale Playbook* | Stanford CS336 systems lectures; GPU MODE lectures |
| Research awareness | Lilian Weng's blog; Sebastian Raschka's *Ahead of AI*; Stanford **CS25** (Transformers United) seminars | Yannic Kilcher paper walkthroughs |

*Video choice:* do Karpathy first (about 15 hours, code-along). Then do CS336 alongside modules 3–12. Skip generic "AI news" channels: reading papers and model cards is faster.

---

## Module 0: setup (half a day)
- Python 3.12, PyTorch, `uv` or venv, Weights & Biases or a CSV logger, and `pytest`.
- Write a 30-line `seed_everything()` and a run-receipt writer (config, git hash, seed, metrics). The GLM repo's `artifacts/receipts` is a good model of this.

## Module 1: autodiff and optimisation from first principles
- **Maths:** reverse-mode AD as a vector–Jacobian product; the softmax + cross-entropy gradient `p − y`; and why Adam ≈ sign-SGD with per-coordinate scaling. Derive AdamW's decoupled weight decay.
- **Code:** a scalar autograd engine (micrograd), then a tensor version with numpy broadcasting. Check every gradient with finite differences.
- **Experiment:** SGD vs Adam vs AdamW on a 2-layer MLP. Plot the loss-vs-LR curve and identify the "edge of stability".

## Module 2: tokenisation and data
- **Maths:** BPE as greedy compression and its link to the entropy rate. Why a byte tokenizer (as in the GLM repo) costs about 4× sequence length.
- **Code:** train a byte-level BPE (Karpathy's `minbpe` style). Measure compression on code vs prose.
- **Experiment:** the same tiny model with byte tokens vs BPE-512 vs BPE-4k. Report *bits per byte*, not loss per token, so the comparison is fair.

## Module 3: the transformer, derived
- **Maths:** attention as kernel smoothing; the `1/√d` scaling (variance argument); pre-norm vs post-norm gradient flow; RMSNorm; SwiGLU; and RoPE as a rotation in 2-D subspaces (prove that `⟨R_m q, R_n k⟩` depends only on `m − n`).
- **Code:** a GPT in under 300 lines: causal mask, KV cache, weight tying, and `torch.compile`.
- **Experiment:** train on TinyStories or code for 10 minutes. Ablate RoPE vs learned positions vs NoPE, using 3 seeds per arm.

## Module 4: training dynamics and scaling
- **Maths:** Kaplan (2020) and Chinchilla (Hoffmann 2022) scaling laws, `C ≈ 6ND`, compute-optimal `N`/`D`; μP (Yang et al.) and why it lets you tune the LR on a small model and transfer it.
- **Code:** LR warmup + cosine/WSD schedule; gradient clipping; mixed precision (bf16); gradient accumulation.
- **Experiment:** fit your own mini scaling law: 5 model sizes × 3 data sizes. Fit `L(N, D) = E + A/N^α + B/D^β` with nonlinear least squares and report CIs on α and β. This is your strongest transferable research skill.
- **Also:** Muon optimiser (orthogonalised momentum via Newton–Schulz) vs AdamW, the most active optimiser topic of 2025–26.

## Module 5: efficient attention (what GLM-5.3-Flash actually uses)
- **Maths:**
  - Linear attention as an RNN: `S_t = S_{t−1} + k_t v_tᵀ`.
  - The delta rule `S_t = S_{t−1}(I − β k_t k_tᵀ) + β k_t v_tᵀ` as online least squares.
  - The gated variants: Gated DeltaNet, then Kimi Delta Attention (KDA).
  - Chunkwise-parallel training (WY representation).
  - Sparse attention: DeepSeek Sparse Attention's lightning indexer + top-k; MLA's low-rank KV compression.
- **Code:** (1) naive softmax attention, (2) recurrent delta rule, (3) chunked delta rule, checking that (2) and (3) give identical outputs; (4) a 3:1 linear:full hybrid stack.
- **Experiment:** associative recall / MQAR synthetic task: pure softmax vs pure linear vs 3:1 hybrid, as a function of sequence length. This reproduces *why* hybrids exist.
- **Reading:** Katharopoulos et al. 2020; Yang et al., *Gated Delta Networks*; *Kimi Linear* tech report (KDA); DeepSeek-V2 (MLA) and V3.2 (DSA); Mamba / Mamba-2 (state-space duality).

## Module 6: mixture of experts
- **Maths:**
  - Top-k routing, capacity factor.
  - The Switch/GShard auxiliary loss `E Σ f_i P_i`, and why it must use the differentiable `P_i`. The GLM repo gets this wrong (review, issue 2).
  - Router z-loss.
  - Fine-grained + shared experts (DeepSeekMoE).
  - Aux-loss-free balancing with per-expert bias (DeepSeek-V3).
- **Code:** a vectorised MoE (sort tokens by expert, no Python loop over experts); a correct aux loss; expert-utilisation entropy logging.
- **Experiment:** dense vs MoE at equal *active* params and equal FLOPs, over 3 seeds. Track expert collapse with and without balancing.

## Module 7: residual-stream research (hyper-connections, mHC)
- **Maths:** the residual stream as a product of layer maps `Π(I + J_l)`; the signal-propagation view; Hyper-Connections (Zhu et al., 2024); manifold-constrained HC (DeepSeek, 2025–26): projecting `H_res` onto doubly-stochastic matrices with Sinkhorn so that `‖H_res‖ ≤ 1` and the identity mapping is preserved.
- **Code:** implement HC and mHC correctly. Start from the GLM repo's `HyperConnection`, fix the double residual, and add Sinkhorn.
- **Experiment:** log per-layer activation and gradient norms for baseline vs HC vs mHC at 12 and 48 layers. The `verify_glm_hyperconnection.py` script in this folder is your starting point.

## Module 8: evaluation and statistics for ML (your PhD advantage)
- Paired designs, exact McNemar, bootstrap over *tasks* (not samples), seed variance, multiple-comparison control (Holm/BH), and minimum detectable effect before you run anything.
- pass@k unbiased estimator (Chen et al. 2021): `1 − C(n−c, k)/C(n, k)`.
- Contamination and leakage checks (the GLM repo's benchmark leaks its reference bodies into pretraining; see review, issue 6).
- **Exercise:** re-analyse the GLM repo's receipts and compute the power of its group-size confirmation. How many tasks would it have needed to detect ±5 pp?

## Module 9: post-training I (SFT, LoRA, preference optimisation)
- **Maths:** LoRA as a low-rank update and why the `α/r` scaling matters; QLoRA (NF4); DPO derived from the KL-regularised RLHF objective (closed-form optimal policy → Bradley–Terry likelihood).
- **Code:** SFT with loss masking on the prompt; LoRA from scratch, then compare with `peft`; DPO in about 50 lines.
- **Experiment:** SFT vs SFT+DPO on a small instruct model (Gemma or Qwen at 1–4B), with a held-out judge plus exact-match tasks.

## Module 10: post-training II (RL with verifiable rewards)
- **Maths:** the policy-gradient theorem; baselines and variance; RLOO; PPO's clipped surrogate; GRPO (DeepSeekMath) and its biases (Dr. GRPO: length and std normalisation); DAPO tricks; importance sampling when the sampling policy ≠ the scoring policy (the GLM repo's issue 3).
- **Code:** fork the GLM repo's `train_rl.py`, fix issues 3 and 4, and add GRPO and Dr. GRPO variants.
- **Experiment:** replicate the repo's temperature ablation with the bug fixed, on ≥5 seeds, with a confirmation split held out. Does T = 0.20 still lead?

## Module 11: inference and systems
- KV-cache arithmetic (bytes per token per layer for MHA, GQA, MLA and linear attention); speculative decoding / MTP; quantisation (int8, int4 / w4a16, QAT vs PTQ); FlashAttention's tiling idea; vLLM paged attention.
- **Exercise:** compute GLM-5.3-Flash's and Gemma-4-31B's KV cache at 128k context from their configs.

## Module 12: agents (feeds directly into the Kaggle competition)
- ReAct; tool calling; SWE-bench and SWE-agent (agent–computer interface design); Agentless (localise → repair → validate); OpenHands; test-time scaling (sample N patches, select by tests / self-consistency); context management for long tasks.
- **Code:** a minimal agent loop: model + 4 tools (`read_file`, `search`, `edit`, `run_tests`) + budget control.
- **Experiment:** see `03_kaggle_gemma4_agent_plan.md`.

---

## Paper reading list (in order)
1. Vaswani et al., *Attention Is All You Need* (2017)
2. Radford et al., GPT-2 (2019); Brown et al., GPT-3 (2020)
3. Su et al., *RoFormer / RoPE* (2021); Shazeer, *GLU Variants* (2020); Zhang & Sennrich, *RMSNorm* (2019)
4. Kaplan et al. (2020); Hoffmann et al., *Chinchilla* (2022); Yang et al., *μP / Tensor Programs V* (2022)
5. Fedus et al., *Switch Transformer* (2021); Dai et al., *DeepSeekMoE* (2024); *DeepSeek-V3 Technical Report* (2024)
6. Katharopoulos et al., *Transformers are RNNs* (2020); Gu & Dao, *Mamba* (2023) and Mamba-2 (2024); Yang et al., *Gated Delta Networks* (2024); Moonshot, *Kimi Linear* (2025)
7. DeepSeek-V2 (MLA) and DeepSeek-V3.2 (DSA sparse attention)
8. Zhu et al., *Hyper-Connections* (2024); DeepSeek, *mHC: Manifold-Constrained Hyper-Connections*
9. Hu et al., *LoRA* (2021); Dettmers et al., *QLoRA* (2023); Rafailov et al., *DPO* (2023)
10. Shao et al., *DeepSeekMath / GRPO* (2024); Ahmadian et al., *Back to Basics: RLOO* (2024); DeepSeek-R1 (2025); Liu et al., *Understanding R1-Zero-like Training / Dr. GRPO* (2025); *DAPO* (2025)
11. Jimenez et al., *SWE-bench* (2023); Yang et al., *SWE-agent* (2024); Xia et al., *Agentless* (2024)
12. Model cards and technical reports to read critically: GLM-5.x, Gemma 4, Qwen3, Kimi K2, DeepSeek-V3.x

## How to "improve things": a research loop
1. Pick **one** component and **one** metric. Write down the hypothesis and the minimum effect you care about *before* running anything.
2. Match the compute between arms (FLOPs or tokens, not steps), use ≥3 seeds, and keep a dev and a held-out confirm split.
3. Screen cheaply on dev → confirm once on held-out with fresh seeds. This is what the GLM repo did, and it caught a false positive.
4. Log negative results. Most ideas fail; the log is what makes your next idea better.
5. Scale up only what survives confirmation, and check that the effect persists at 2–4× scale.
