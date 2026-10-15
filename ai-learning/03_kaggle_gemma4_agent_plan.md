# Kaggle: Google – The Gemma 4 Developer Agent Competition: experiment plan

> Details below come from web search on 2026-10-15. Kaggle itself was not reachable from this sandbox, so **confirm everything on the competition's Overview, Data and Rules pages**. In particular, sources disagree on the entry deadline (Nov 2 vs Nov 25, 2026).

## What the competition is (as reported)
- A **SWE-bench-style** task: your agent gets a repository plus an issue, and must produce a patch that passes hidden validation tests. Score = % of tasks whose patch passes.
- You submit an **agent config zip** with `agent.yaml` at its root. It follows the **Google ADK Agent Config** spec, with restrictions so that no code runs outside the sandbox. You can ship **system prompts, custom tools, skills and LoRA adapters**. You do not ship arbitrary code.
- Model: reportedly only `gemma-4-31b-it-qat-w4a16-ct` (Gemma 4 31B, QAT, 4-bit weights).
- Budget: **12 hours for all tasks** (including sandbox setup). The hidden set is reportedly about 120 tasks from private repos, split into public and private leaderboards.
- Prizes: $65k main (37k / 18k / 10k), plus a separate $35k paper track.

So the levers you control are **prompting, tool and skill design, the agent's control flow (as far as ADK config allows), test-time compute allocation, and LoRA fine-tuning**. Pretraining-style work from the GLM repo does not apply directly. The RL, evaluation and statistics lessons do.

## The statistical reality (read before tuning anything)
With about 60 tasks on the public leaderboard, a 30% solve rate has a standard error of `√(0.3·0.7/60) ≈ 5.9 pp`. **Leaderboard differences under about 10 pp between two submissions are mostly noise.** You need your own local evaluation set and paired comparisons. This is exactly the trap the GLM repo fell into and then caught: its dev winner reversed on fresh seeds.

## Plan

### Phase 1: harness (days 1–3)
1. Build a local evaluation set of 100–300 tasks that resembles the hidden set: SWE-bench Verified / Lite, SWE-Gym or SWE-smith instances. Prefer repos *not* heavily represented in pretraining, to limit contamination. Split them into **dev (screening)** and **confirm (opened rarely)**.
2. Run the exact submitted config locally, with the same model and quantisation (w4a16) and the same tool restrictions. Log every trajectory: tokens, tool calls, wall time, and the final patch.
3. Baseline: the starter config. Record the solve rate with a bootstrap CI over tasks, plus time per task. **Time is a hard constraint:** 12 h / 120 tasks ≈ 6 min per task on average.

### Phase 2: cheap, high-leverage ablations (one factor at a time, paired on the same tasks)
| Lever | Hypothesis | Metric |
|---|---|---|
| Localisation-first workflow (Agentless style: search → rank files → read → edit) | Small models fail mostly at *finding* the bug | solve rate; % of trajectories touching the gold file |
| Tool interface (SWE-agent ACI: windowed file viewer, line-anchored edit, lint-on-edit) | Fewer malformed edits | invalid-edit rate |
| "Reproduce first" skill: write a failing test, then fix | Better patch validity | solve rate |
| Test-time sampling: N candidate patches, pick by tests / majority | +pp at fixed budget | solve rate per GPU-hour |
| Budget controller: early stop, max steps, context truncation and summarisation | Avoid timeouts | timeout rate |
| System prompt variants | Format adherence | invalid-action rate |

Analyse each pair with **exact McNemar on per-task pass/fail**, Holm-corrected across arms. Only promote arms that win on **confirm**.

### Phase 3: LoRA fine-tuning (after the agent loop is solid)
- Data: successful trajectories (rejection sampling). Run the current agent on many training tasks and keep the passing ones. Optionally add trajectories from SWE-Gym / SWE-smith datasets, *formatted exactly like your tool schema*.
- Method: SFT with loss on assistant/tool-call tokens only. Then optionally run one round of rejection-sampling fine-tuning (expert iteration). This is the cheap cousin of the RLOO in the GLM repo.
- Check: does the LoRA survive the **w4a16 QAT** base? Train against the quantised base, or evaluate the merged/adapter model under the exact competition quantisation.
- Watch for: overfitting to one repo's style, and regressions on tasks that the base model solved (track per-task wins *and* losses, as the GLM repo did per family).

### Phase 4: paper track
Your edge is rigour. A paper on *"Which agent design choices matter for a 31B 4-bit open model, under a fixed time budget — with paired statistics and a replication split"* fits the paper track's judging well. Keep the experiment log from day 1 so the paper writes itself.

## Links to the GLM review
- Issue 3 (sampling/scoring temperature mismatch) is the same class of bug as training a LoRA on trajectories sampled with a different prompt or temperature than you deploy with. Keep train and serve conditions identical.
- Issue 6 (leaky benchmark): make sure your local dev tasks do not overlap your LoRA training tasks at the *repo* level, not just the task level.

## Sources
- [Google – The Gemma 4 Developer Agent Competition (Kaggle)](https://www.kaggle.com/competitions/gemma-4-developer-agent)
- [Gemma 4 Developer Agent Paper Track (Kaggle)](https://www.kaggle.com/competitions/gemma-4-developer-agent-paper)
- [Google Gemma announcement on X](https://x.com/googlegemma/status/2104619611417391289)
- [Community notebook: "Gemma 4: Measure Before You Tune"](https://www.kaggle.com/code/dmitriigluzdov/gemma-4-measure-before-you-tune)
- [Community notes: sweeden-ttu/developer-agent](https://github.com/sweeden-ttu/developer-agent)
