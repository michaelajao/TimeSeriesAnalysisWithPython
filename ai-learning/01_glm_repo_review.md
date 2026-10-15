# Review: `vukrosic/glm-5.3-flash-from-scratch`

Reviewed at commit `96d9c7c` (2026-08-28). I read all of the model, training, RL, evaluator and slide code. PyTorch could not be installed in this sandbox, so I did not re-run training. Every finding below comes from reading the code. Where a check could be done without PyTorch, it was done in numpy (`verify_glm_hyperconnection.py`).

## Verdict

**It is a good, honest teaching repo. It is not a faithful GLM-5.3-Flash, and it has a few real bugs.** You can learn from it. Don't copy its "advanced" parts (hyper-connections, MoE balancing, RL objective) into your own work without the fixes below.

What it does well:
- It states its own scale-down honestly (25.7M vs 320B params, byte tokenizer, 192-token context). The headline specs it quotes for the real model match public sources: 45 layers (34 KDA linear + 11 sparse), 288 routed experts with top-8, 1 shared expert, 4-stream mHC, 1M context, 320B total / 18B active.
- Experiment hygiene is better than most YouTube repos. It has frozen dev/confirm splits, checkpoint selection on dev only, paired exact McNemar tests, Holm correction, a task bootstrap, and a replication on fresh seeds. It even reports that its own dev "winner" (group size 4) **failed** to replicate. That is the most valuable lesson in the repo.
- The receipts (hashes, seeds, every rollout) make the claims auditable.

## Correctness issues (ordered by importance)

| # | Where | Issue | Effect | Fix |
|---|---|---|---|---|
| 1 | `glm53_flash/model.py` `HybridBlock.forward` + `HyperConnection.forward` | **Double residual.** The block returns `x + attn + ffn`, which already includes its input. The hyper-connection then does `streams + routed * transformed`, adding the input a second time. The effective residual map is `I + r·aᵀ`, a non-contractive matrix. mHC exists precisely to prevent this: it constrains the residual mixing to be doubly stochastic. | Hidden-state norm grows ×(1.25)¹² ≈ **14.5×** across 12 layers at init (verified in numpy). Pre-norm hides it in the forward pass, but gradient scale and the "four streams" are not doing what the slides describe. | Make the block return only the delta (`attn_out + ffn_out`), or keep the block residual and drop the outer add. For real mHC: `x_{l+1} = H_res x_l + H_postᵀ F(H_pre x_l)` with `H_res` projected by Sinkhorn. |
| 2 | `SparseMoE.forward` + `train_pretrain.py` balance loss | **The load-balancing loss has no gradient.** `usage` is filled from `positions.numel()`, an integer count, so the `0.01 * balance` term is a constant and does nothing. The router only learns through the top-k softmax weights, so expert collapse is unprevented. | Logged "router_balance_loss" looks like regularisation but isn't. | Use the Switch/GShard loss `E · Σ_i f_i · P_i`, where `P_i` is the mean router softmax probability (differentiable), or DeepSeek-V3's aux-loss-free bias update. |
| 3 | `runtime.generate_group` vs `train_rl.completion_log_probabilities` | **Sampling/scoring mismatch.** Rollouts are sampled from `softmax(logits / T)` with T = 0.35, and pad/BOS/SEP are masked out. The policy gradient uses `log_softmax(logits)` at T = 1 with no mask. REINFORCE/RLOO is then off-policy without an importance correction, so the gradient is biased. | It helps explain why the temperature ablation behaves oddly. The repo's "strongest lead" (T = 0.20) is confounded by this. | Score with the same temperature and mask you sampled with, or add an importance ratio. |
| 4 | `completion_log_probabilities` | Log-prob is **divided by completion length**. Combined with the −0.1 invalid penalty, this produces the length bias described in *Understanding R1-Zero-like Training* (Dr. GRPO, 2025): short wrong answers are punished more per token than long wrong ones. | Minor at this scale. | Use a sum, or a constant normaliser. |
| 5 | `LinearAttention` | RoPE is applied **before** the ELU+1 feature map, which destroys RoPE's relative-position property. The prefix state is materialised as `[B,T,H,D,D]` (O(T·D²) memory). | Fine at 192 tokens but neither faithful nor scalable. GLM uses KDA (Kimi Delta Attention, a gated delta rule), not ELU linear attention. The README admits this. | For learning, implement the chunked delta rule (Gated DeltaNet / KDA papers). |
| 6 | Benchmark design (`tasks.py`) | Pretraining text contains the **exact reference bodies** for all 8 families. Dev and confirm reuse the same bodies and the same 3 test cases; only the function name and one of two descriptions change. | "Unseen" means only unseen names. RL is sharpening a memorised template, not learning to code. The README says it "does not establish broad coding ability", which is correct. | For your own experiments, hold out operations or compositions, not just names. |
| 7 | `slides/slides.md` | The title slide references `assets/glm53-official-architecture.png`, which is not in the repo. | A broken image on slide 1 if you build from the markdown. | — |
| 8 | `evaluator.py` | An AST deny-list plus restricted builtins is fine for this toy grammar. It is **not** a sandbox for real model code: there is no timeout or memory limit, and it runs in-process `exec`. | Don't reuse it for Kaggle or real agent work. | Use a subprocess/container with time and memory limits. |

Things I checked that are fine: the causal masking in the gathered sparse attention; tied embeddings; the RoPE rotation itself; RLOO's leave-one-out baseline; the McNemar p-value (16 wins / 0 losses → 2·0.5¹⁶ = 3.05e-5 ✓); and the pad masking in pretraining (`-100`).

## How the slides were built (evidence)

- **There is no slide framework.** `slides/slides.html` is one 223 KB, hand-authored single file of plain HTML, CSS and vanilla JS: inline SVG animations, CSS keyframes, and arrow-key navigation. It has no external `<script src>`. The only external links are source citations. It does not use Reveal.js, Marp, Slidev or D3.
- `slides/slides.md` is a markdown source with `<!-- slide: … -->` markers and `---` separators, mixed with raw HTML/SVG/JS blocks. No converter is committed, so the HTML was probably generated from or kept in sync with that file outside the repo.
- `slides/serve_slides.py` is a small Python `http.server` that adds a **live slide-feedback/text-editor**. Edits and notes are saved to `slide-feedback.json` / `slide-text-edits.json` and `localStorage`, and there is a "recording mode" toggle (commit `ab8129a`) that hides the editing UI for screen recording.
- `slides/slides.html.review.md` ("STATUS: PASS · Slides: 76 · Rendered all 76 slides in recording mode…") is the output of an automated visual QA pass. It renders every slide in a headless browser and checks for clipping.
- **Inference (not proven):** the commit pattern suggests the deck was written by an AI coding agent driven by the author's per-slide feedback. That pattern is dozens of tiny commits like "Clarify…", "Put plain language stream explanation on slide", an autosaving feedback editor, and a machine-style review report. The colour palette (`#d97757` on `#f4f1e8`) matches Claude's brand colours. Nothing in the repo names the tool.

**To build the same kind of deck yourself:** write `slides.md`, ask an agent to render it into one HTML file with CSS animations, serve it with a tiny feedback server, iterate, then screen-record in recording mode. If you'd rather use a framework, Slidev (markdown + Vue) or Reveal.js give you the same result with less custom JS.

## What I could not check
- **The YouTube video.** The sandbox cannot reach youtube.com, so this review covers the repo, not the narration.
- **Your Zotero library.** This session runs in a cloud container, not on your computer. To include it, export it (Zotero → select collection → right-click → *Export Collection…* → BibTeX or CSV, or *Generate Report*). Then commit the file into this repo or paste it, and I'll map your textbooks onto the curriculum.
