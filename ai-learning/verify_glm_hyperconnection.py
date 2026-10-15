"""Numpy check of the residual path in vukrosic/glm-5.3-flash-from-scratch.

In that repo, HybridBlock.forward returns `x + attn + ffn` (it already contains
its own residual), and HyperConnection then adds that whole output back onto the
streams: `streams + transformed * routed`. The input is therefore counted twice
per layer. This script mirrors the stream arithmetic at initialisation (sublayer
outputs ~0, as with std=0.02 weights) and compares it with a block that returns
only the sublayer delta, which is what hyper-connection papers assume.
"""
import numpy as np

STREAMS, LAYERS, DIM = 4, 12, 192
rng = np.random.default_rng(0)


def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


input_w = softmax(np.linspace(0.1, -0.1, STREAMS))   # repo's input_logits init
output_w = softmax(np.linspace(-0.1, 0.1, STREAMS))  # repo's output_logits init


def run(block_returns_residual: bool, sublayer_scale: float) -> float:
    x0 = rng.standard_normal(DIM)
    streams = np.tile(x0, (STREAMS, 1))
    for _ in range(LAYERS):
        mixed = input_w @ streams
        delta = sublayer_scale * rng.standard_normal(DIM)  # stands in for attn + MoE
        transformed = mixed + delta if block_returns_residual else delta
        streams = streams + output_w[:, None] * transformed[None, :]
    final = streams.mean(axis=0)
    return np.linalg.norm(final) / np.linalg.norm(x0)


if __name__ == "__main__":
    for scale in (0.0, 0.05):
        repo = run(True, scale)
        fixed = run(False, scale)
        print(f"sublayer scale {scale}: |h_final|/|h_0| repo-style = {repo:6.2f}   delta-only = {fixed:5.2f}")
    print(f"analytic (scale 0): (1 + 1/4)^{LAYERS} = {(1.25) ** LAYERS:.2f}")
