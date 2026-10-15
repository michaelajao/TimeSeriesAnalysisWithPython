# Second opinion on the Gemma 4 agent plan + compute allocation

This reviews the plan the user pasted from another assistant session on 2026-10-15. That session had read `gemma4-developer-agent` on the user's Windows machine. **This session cannot see that repo** (it runs in a cloud sandbox), nor the Coventry-hosted Sulis wiki (`github.coventry.ac.uk` is blocked from here). Facts about the repo (F57, F58, F61, F63, the 28/68 and 30/68 scores, the token rates) are taken from that paste and have not been re-checked.

## 1. What I'd keep, change or add

### Keep
- **Sulis L40 as the timing proxy.** The Kaggle scorer reportedly uses 4× L4. The L4 and L40 are both Ada Lovelace (sm_89), so they use the same kernels, and the measured 28.2 vs 30.9 tok/s calibration makes sense.
- **The generation-budget sweep (E1).** If 26 of 68 tasks hit the 8-minute cap and 99% of wall time is generation, the thinking budget is the biggest lever on the board.
- **The trace-driven approach.** F57/F58-style accounting of where minutes go is exactly right.

### Correct
1. **The scratch-file fix will not recover "~10% of generation time".** Per F57, malformed edits *and* rejected writes *together* cost 43/416 and 53/432 minutes (~10–12%). The `/tmp` rejections are only part of that. Expect a few percent from this fix alone, and measure it as *rejected `write_file` calls → 0*, not as solve rate.
2. **The here-doc fix may run into the same JSON escaping bug.** F57 says Gemma's tool-call JSON mangles escaped quotes. A `cat << 'EOF' … EOF` sent through `run_command` is still a JSON string argument, so Python code full of quotes can break the same way. Ranked options:
   - (a) Say in the prompt: *"to run a throwaway check, use `run_command` with `python - <<'EOF' … EOF`; do not save it"*. Nothing lands on disk, and the model learns one pattern.
   - (b) If `sredit` is pre-installed, reuse its here-doc transport, which is already proven.
   - (c) Check how the final diff is computed. If it is `git diff` over tracked files only, a git-ignored `/workspace/.scratch/` would be safe, which removes the problem at its source. **Verify before relying on it:** if the harness does `git add -A`, scratch files will pollute the patch.
3. **"Arm 3 adaptive thinking (4096 on the first call, 1024 afterwards)"** may not be expressible in an ADK agent config, which forbids running code outside the sandbox. Check that `sampling.yaml` and ADK support per-turn budgets before you spend a Sulis run on it. If they don't, run Arms 1 and 2 only.
4. **The `sbatch --array=0-3` command shards one arm 4 ways.** It does not run 4 arms. That's fine, but each arm needs its own submission with the same `TASKS_FILE` and seeds, so the comparison stays paired.

### Add (missing from the pasted plan)
- **Power.** On 68 paired tasks with one run per arm, exact McNemar needs roughly these splits for p < 0.05 (computed here):

  | discordant tasks | needed win–loss split | net gain |
  |---|---|---|
  | 10 | 9–1 | +8 |
  | 12 | 10–2 | +8 |
  | 15 | 12–3 | +9 |
  | 20 | 15–5 | +10 |

  So a single run detects only gains of about **+12–15 pp**. Your own v5 scored 28 and then 30 with no change, so run-to-run noise alone produces discordant tasks. Implications:
  - **Cheap mechanical fixes** (scratch writes, edit transport) should be accepted on their *mechanism metric* (rejected calls, malformed edits, timeouts) plus "no significant regression". Don't wait for solve-rate significance.
  - **Real design choices** (thinking budget, multi-file workflow) need 2 runs per arm. Analyse them with a per-task model, e.g. logistic GLMM with a task random effect, or pooled McNemar over task × run.
  - **Use racing / successive halving** across variants: drop losers after a 34-task half-run. Your metaheuristics books cover this (F-Race, Talbi ch. on noisy optimisation). It's the right way to spend Sulis hours.
- **Multi-file tasks are the largest headroom.** 1/19 solved means 19 of 68 dev tasks (28%) are nearly all lost. But a prompt that forces "enumerate files, touch all of them" can hurt single-file tasks. Always report the effect **separately for single-file and multi-file subsets**.
- **Combining `sredit` with `contract_first_complete`.** `sredit` alone scored 25/68, below v5's 28 and 30. The combination is worth one run, but treat it as a new arm, not a guaranteed sum of gains.
- **LoRA is still unused.** Submissions may ship LoRA adapters. Rejection-sampled successful trajectories from your own dev-excluded tasks → SFT LoRA (on Sulis A100s) targets the two biggest failure modes directly: malformed tool JSON and runaway thinking. Train against the *same quantised base* you submit with, and keep the training repos disjoint from the dev repos.
- **Deadline.** Search results disagree on the entry deadline (Nov 2 vs Nov 25, 2026). If it is Nov 2, rule acceptance and team merger have to happen this month. Check the Rules tab today.

## 2. Compute allocation (corrected)

| Machine | Use it for | Watch out for |
|---|---|---|
| Laptop (T600 4 GB) | Packaging, trace analysis, prompt edits, statistics | — |
| Home RTX 5060 Ti 16 GB (Blackwell, sm_120) | Tool and harness dry-runs with small stand-in Gemma models; LoRA data preprocessing | Consumer Blackwell needs recent CUDA 12.8+/vLLM builds; 31B w4a16 won't fit with a useful KV cache |
| **Brosnan, 2× Quadro RTX 8000 48 GB (Turing, sm_75)** | Interactive debugging of tool behaviour and trace inspection | **Not a timing proxy, and maybe not even a faithful runner.** Turing has no bf16, and vLLM's fast w4a16 Marlin kernels need Ampere (sm_80) or newer, so the exact competition checkpoint may fall back to slow kernels, fail to load, or run in fp16. Gemma models have a history of fp16 activation overflow. Validate outputs against Sulis on a few tasks before trusting any Brosnan behaviour. |
| **Sulis L40 46 GB** (Ada, sm_89) | All solve-rate and timing experiments (same architecture as the Kaggle L4) | Queue time: use racing so arms that are clearly losing stop early |
| **Sulis A100** | LoRA SFT training; bulk trajectory generation for rejection sampling | A100 timing ≠ L4 timing; never use it for budget calibration |
| Kaggle (1 submission/day) | Use the daily slot for the *current best confirmed* bundle; never for screening | Public LB ≈ 60 tasks, SE ≈ 6 pp |

## 3. Paper track ("it doesn't have to be graph-based")

Agreed: **don't force a graph method.** Nothing in your log shows a graph-based gain, and five weeks is too short to build and validate one. The strongest paper is the one your traces already support. I'd **merge** the two pasted options into one paper:

> **Where the minutes go: generation budgets and tool-call serialization failures in a 4-bit 31B open coding agent under a fixed wall-clock limit**
>
> 1. Time accounting: generation is >98% of wall time; tool time is <2%. The 4-way concurrency slowdown (1.6×); L4 vs L40 token-rate calibration.
> 2. Failure taxonomy from traces: runaway thinking (F58), malformed JSON edits (F57), rejected writes, and multi-file incompleteness (F63).
> 3. Interventions, each paired and pre-registered on dev and confirmed once on a held-out split: thinking budget, edit transport (JSON `edit_file` vs here-doc `sredit`), scratch-write policy, multi-file checklist.
> 4. Statistics: paired exact tests, a task random-effects model, and the power table above. Be explicit about the replication run (28 → 30) as a noise estimate.

This is a measurement paper with transferable lessons for anyone serving open-weight models through vLLM. That plays to your strengths (rigour, statistics), not to novelty you would have to rush.

## 4. Order of work (next 5 days, up to the 20 Oct gate)
1. **Today:**
   - Apply the scratch-file prompt fix, using option (a) above.
   - Check how the diff is computed (option c).
   - Package the bundle and use the daily Kaggle slot on v5 + fix.
   - Confirm the entry deadline.
2. **Sulis, days 1–2:** thinking budget 4096 vs 2048, two runs each, on the same 68 tasks. Record cap hits, malformed calls and solve rate, split by single- and multi-file tasks.
3. **Sulis, days 2–3:** v5 + `sredit` pre-installed vs v5. Then the multi-file checklist arm.
4. **A100 in parallel:** generate trajectories on non-dev repos for a LoRA dataset.
5. **20 Oct gate:** lock the paper outline (above) and freeze the confirm split.
