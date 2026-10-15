# Your Zotero library mapped onto the curriculum

Source: the `Zotero_Report.htm` export (57 items). Module numbers refer to `02_curriculum.md`.

## Use these as primary texts
| Your book | Modules | How to use it |
|---|---|---|
| Bishop & Bishop, *Deep Learning: Foundations and Concepts* (2024) | 1, 3, 4, 9 | **Your main textbook.** Chapters on backprop, normalisation, transformers, and generative models. Do the exercises in the transformer chapter before writing Module 3 code. |
| MacKay, *Information Theory, Inference, and Learning Algorithms* | 2, 8 | Entropy, compression and arithmetic coding give you the language-modelling = compression view, and the reason to report bits per byte. Bayesian model comparison feeds into the Module 8 statistics. |
| *Mathematical Aspects of Deep Learning* (Grohs & Kutyniok, CUP 2022) | 4, 7 | Signal propagation, expressivity and optimisation-landscape chapters. These are the theory behind the residual-stream (HC/mHC) module. |
| Kochenderfer & Wheeler, *Algorithms for Optimization*; Bierlaire, *Optimization: Principles and Algorithms* | 1, 4 | Momentum, adaptive methods, and Newton / quasi-Newton. Newton–Schulz iteration is the core of the Muon optimiser. |
| Brunton & Kutz, *Data-Driven Science and Engineering* | 5 | Linear dynamical systems, DMD and Koopman views. Linear attention and state-space models (Mamba, DeltaNet) **are** learned linear dynamical systems, so this is your bridge from your modelling background. |
| Boguslavskiy, *Dynamic Systems Models: Parameter and State Estimation* | 5 | The delta rule `S_t = S_{t−1}(I − βkkᵀ) + βkvᵀ` is LMS / online least squares. Compare it with RLS and Kalman updates; that is exactly the gated-delta-net insight. |
| Raschka et al., *Machine Learning with PyTorch and Scikit-Learn* | 0, 3 | PyTorch fluency and its transformer chapter. |
| Babcock & Bali, *Generative AI with Python and PyTorch* (2025) | 3, 9 | Practical LLM fine-tuning chapters. |
| Iusztin, *LLM Engineer's Handbook* (2024) | 9, 11, Kaggle | SFT/LoRA pipelines, inference serving, evaluation. Directly useful for the Gemma LoRA work. |
| Raieli & Iuculano, *Building AI Agents with LLMs, RAG, and Knowledge Graphs* (2025) | 12, Kaggle | The agent-loop and tool-design chapters. |
| Talbi, *Metaheuristics*; Jin et al., *Data-Driven Evolutionary Optimization*; Bozorg-Haddad et al. | Kaggle | **Agent-config search is noisy black-box optimisation.** Use racing, successive halving and surrogate-assisted search to allocate Sulis runs across prompt/tool variants. Evolutionary prompt optimisation (e.g. GEPA- or DSPy-style) is the same idea. |
| Hennig, Osborne & Kersting, *Probabilistic Numerics* | 4, Kaggle | Bayesian optimisation for hyperparameters (LR, thinking budget) when each evaluation costs GPU-hours. |

## Useful side tracks
- **Labonne, *Hands-On Graph Neural Networks*:** only if you later try graph-based code localisation (repository call or import graphs). It isn't needed for the current paper.
- **Somani et al., *Interpretability in Deep Learning*; Masís, *Interpretable ML with Python*:** an extension to Module 7. Logit lens and activation patching on the residual stream.
- **Time series (Mills; Manu Joseph, *Modern Time Series Forecasting with Python*; Vandeput):** a natural capstone that links back to this repo. Build a small decoder-only *time-series foundation model* (TimesFM / Chronos style: patching, or tokenising values into bins) with the Module 3 code.
- **Epidemiology and networks** (Kuhl; Brauer et al.; Kiss et al.; Barrat et al.; De Domenico): your domain strength. Physics-informed or LLM-assisted epidemic modelling is a strong research niche where you already hold the domain knowledge.

## Low priority for this goal
Cormen (CLRS), Taha, Williams, the OR encyclopedias and handbooks, *NLP: Semantic Aspects* (2013, pre-transformer), *Text Mining with ML* (2019), the healthcare-analytics titles, and the Julia books. They're good books, just not on the LLM path.

## Gaps worth filling (all free online)
- Prince, *Understanding Deep Learning* (2023): the clearest modern derivations, a companion to Bishop.
- Jurafsky & Martin, *Speech and Language Processing*, 3rd ed. draft: tokenisation, LMs, and evaluation.
- Sutton & Barto, *Reinforcement Learning: An Introduction*, 2nd ed.: needed for Module 10 (policy gradients, baselines). Your library has no RL text.
- Google DeepMind, *How to Scale Your Model*, and Hugging Face, *The Ultra-Scale Playbook*: systems and scaling (Module 11).
- Raschka, *Build a Large Language Model (From Scratch)* (paid): the closest book-length version of Modules 2–3 and 9.
