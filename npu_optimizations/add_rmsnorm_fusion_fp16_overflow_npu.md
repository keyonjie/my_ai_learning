
# Why Add + RMSNorm Fusion Fixes FP16 Overflow on NPUs
## Purpose
This document explains why and how operator fusion (specifically Add + RMSNorm) is used in NPU compilers to eliminate FP16 overflow issues while maintaining—or even improving—performance. It is intended for compiler, kernel, and model engineers who need a clear numerical and performance rationale for this optimization.


## 1. Background: FP16 Overflow in Modern LLM Blocks
In transformer architectures, RMSNorm is almost never applied to a standalone tensor. Instead, it operates on a residual sum:
```python
Y = RMSNorm(X + R)
```
Where:

- X is the output of an attention or MLP block
- R is the residual stream accumulated across layers
### Why FP16 Is Fragile Here

- FP16 has a limited dynamic range (~6.5e4)
- Residual streams grow in magnitude as depth increases
- Activation outliers are common (attention spikes, heavy tails)

If (X + R) is computed and materialized in FP16 before normalization, overflow or severe rounding can occur before RMSNorm has any chance to stabilize the values.


## 2. What Goes Wrong Without Fusion
```python
Unfused Execution (Conceptual)
Add kernel:
  FP16 X + FP16 R -> FP16 T

RMSNorm kernel:
  read T -> square -> reduce -> normalize
```


### Failure Modes

#### Residual add overflow
- X + R may overflow to inf in FP16
- This happens before normalization
#### Poisoned RMS statistics
- A single inf or NaN corrupts mean(x^2)
- The entire vector output becomes invalid
#### Excess FP16 materialization
- Each kernel boundary forces rounding and clipping
- Numerical error accumulates across layers

Once overflow occurs, RMSNorm cannot recover correctness.


## 3. What Fusion Really Means in an NPU Compiler
Fusion is not just a performance optimization—it is a correctness optimization.
When an NPU compiler fuses Add + RMSNorm, the expression (X + R) is treated as an internal value and is never materialized as an FP16 tensor.
### Typical Fused Kernel Behavior
```python
load X (FP16)
load R (FP16)

TMP = X + R              # FP32 or widened accumulator
ACC += TMP * TMP         # FP32 accumulation
RMS = rsqrt(mean + eps)  # FP32
OUT = TMP * RMS * gamma  # FP32

store OUT -> FP16

```

Key properties:

- No FP16 tensor ever holds X + R
- Squaring and reduction use widened precision
- Only the final normalized output is cast to FP16

This removes the primary FP16 overflow window.


## 4. Why RMSNorm-Only Fusion Is Not Enough
If only RMSNorm is fused but the residual add happens outside the fused kernel:
```python
T = X + R        # FP16, materialized
Y = RMSNorm(T)  # fused
```
Overflow may already have occurred at T = X + R.
### Conclusion:
To be numerically safe in FP16, Add and RMSNorm must be fused together.


## 5. Why This Is Especially Important on NPUs
NPUs differ from GPUs in several important ways:

- FP16 is the primary fast path
- Graph-level FP32 execution is often slow or limited
- Accumulator widening is typically kernel-local, not graph-wide
- Tensor materialization and memory traffic are expensive

As a result, NPUs rely on fused kernels to:

- Widen precision only where it matters
- Avoid unnecessary memory writes
- Preserve numerical correctness without switching the entire graph to FP32


## 6. Performance Impact of FP32 Math Inside Fused Kernels
A common concern is that FP32 math inside fused kernels might reduce performance.
### In Practice, It Usually Does Not
Reasons:

- RMSNorm is memory-bound, not compute-bound
- FP32 accumulation is already supported in NPU datapaths
- Fusion removes extra memory traffic and kernel launches
### Comparison

| Implementation                         | Performance       | Stability |
|----------------------------------------|-------------------|-----------|
| Unfused FP16 Add + RMSNorm              | Worst             | Poor      |
| Fused RMSNorm only                      | Medium            | Risky     |
| **Fused Add + RMSNorm (FP32 internal)** | Best or near-best | Stable    |
| Full FP32 graph                         | Slow              | Stable    |		

In many cases, fused kernels with FP32 internal math are faster overall than unfused FP16 implementations due to reduced memory bandwidth and launch overhead.


## 7. Key Mental Model
Normalization fixes scale, not overflow.
Overflow prevention must happen before:

- Squaring
- Reduction
- Division

Add + RMSNorm fusion ensures:

- Residual addition occurs at safe precision
- RMS statistics reflect the true signal
- FP16 is used only for storage, not critical math


## 8. Practical Takeaways

- FP16 overflow in LLMs is often caused by residual adds, not RMSNorm itself
- RMSNorm-only fusion is insufficient
- Add + RMSNorm fusion is a correctness optimization
- FP32 internal math inside fused kernels has negligible performance cost on NPUs
- This pattern delivers FP32-like stability with FP16 throughput


## 9. Recommended Guidance for NPU Compiler and Model Teams

- Always fuse residual add with normalization when targeting FP16
- Prefer widened accumulators inside fused kernels
- Avoid materializing intermediate FP16 tensors for (X + R)
- Treat norm fusion as mandatory for correctness, not an optional optimization


# Summary
Add + RMSNorm fusion closes the most dangerous FP16 overflow window in transformer blocks. On NPUs, it is one of the most effective techniques for achieving both numerical stability and high performance without falling back to full FP32 execution.
