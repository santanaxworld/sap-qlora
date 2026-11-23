# SAP-QLoRA

**$W_0^T$-Free, AVX-512 VNNI Accelerated Fine-Tuning on CPU**

![C++](https://img.shields.io/badge/Language-C%2B%2B-blue)
![SIMD](https://img.shields.io/badge/Kernel-AVX--512%20VNNI-orange)
![Platform](https://img.shields.io/badge/Tested%20On-Ryzen%207%207800X3D-red)

---

### About Me

Hey everyone, I'm Berkay. I'm 21 and a 2nd-year software engineering student.

First of all, my English is not very good, so I use a translator while writing my posts. Also, I am not someone who works on AI full-time or even part-time. I have just been into low-level stuff for a long time (really long :D). Back in 2017, I started playing around with tcpip delayed acks and lowering window sizes to get an advantage in pvp games, and I actually got pretty far with it. Since 2019, I have also been doing reverse engineering.

Basically, I am kind of obsessed with optimization and finding and fixing problems, so that is what I focused on in this project. My next post will probably be about a hypervisor-based (ring -1) api tracer.

Also, I used Gemini Neon for brainstorming, learning, and figuring out the logic, and I did the coding part with Grok 4.7 High. At the very bottom, I will explain in as much detail as I can why I used intrinsic functions and what they actually do in this setup. The whole project took around 13 hours.

Please let me know if you have any feedback or suggestions so I can keep improving!

---

## Overview

**SAP-QLoRA** is a high-performance CPU training system written from scratch in C++. In frozen-weight low-rank fine-tuning (LoRA / QLoRA), it completely removes the $W_0^T$ matrix multiplication in the backward pass using the **Self-Aligning Projection (SAP)** algorithm, and runs the forward pass directly with an **INT8 AVX-512 VNNI** kernel.

While standard QLoRA methods give up training speed just to save memory, **SAP-QLoRA** both shrinks $W_0$ memory by **4×** and gives a **3.70×** speedup over FP32 `BP-LoRA`, and a **2.31×** speedup over a fair single-copy INT8 `BP-QLoRA` at the $D = 2048$ scale.

### Test Machine Specs
* **CPU:** AMD Ryzen 7 7800X3D
* **RAM:** Corsair 32GB (16x2) Vengeance 6000MHz CL36 DDR5

### Key Results at a Glance

| Metric | Measured Result | Reference / Note |
|---|---|---|
| **Total Speedup ($D = 2048$)** | **3.70×** (`1240.48 → 335.52 ms/step`) | vs FP32 `BP-LoRA` |
| **Pure SAP Speedup ($D = 2048$)** | **2.31×** (`775.31 → 335.52 ms/step`) | vs fair INT8 `BP-QLoRA` (backward pass is **5.6×** faster) |
| **VNNI Speedup ($D = 2048$)** | **1.60×** (`1240.48 → 775.31 ms/step`) | `BP-LoRA → BP-QLoRA` forward pass gain |
| **$W_0$ Memory Savings** | **4.0×** (`96.00 → 24.09 MB`) | Constant across all $D$ scales |
| **Total Memory Savings ($D = 2048$)** | **-28%** (`247.43 → 179.12 MB`) | Only `+2.0%` more than `BP-QLoRA` (`175.52 MB`) |
| **Accuracy Gap ($D = 256, L = 6$)** | **`+0.034 BPC`** | `2.607` vs `2.641` (2000 steps) |
| **Accuracy Gap ($D = 1024, L = 6$)** | **`+0.036 BPC`** | `2.940` vs `2.975` (same warmup+cosine schedule) |
| **Depth Test ($D = 256, L = 12$)** | **`+0.030 BPC`** | `2.809` vs `2.839`; $\|\delta_0\|/\|\delta_{11}\| = 1.89$ |
| **Time-to-Accuracy ($D = 1024$)** | **~2.4× faster** | Reaching `3.00 BPC`: `42 s` vs `104 s` |

---

## Table of Contents
1. [Why Does Normal QLoRA Slow Things Down?](#1-why-does-normal-qlora-slow-things-down)
2. [Architecture and Algorithm](#2-architecture-and-algorithm)
3. [Test Setup and Compared Modes](#3-test-setup-and-compared-modes)
4. [Speed, Memory, and Breakdown Results](#4-speed-memory-and-breakdown-results-d--256-to-2048)
5. [Accuracy (BPC) and Ablation Studies](#5-accuracy-bpc-and-ablation-studies)
6. [Time-to-Accuracy ($D = 1024$)](#6-time-to-accuracy-d--1024)
7. [Depth Test ($D = 256, L = 12$)](#7-depth-test-d--256-l--12)
8. [Honest Limits and Scope ($V \ll D$ Regime)](#8-honest-limits-and-scope-v-ll-d-regime)
9. [Discarded Ideas (Negative Results Log)](#9-discarded-ideas-negative-results-log)
10. [Why Intrinsics?](#10-why-intrinsics)

---

## 1. Why Does Normal QLoRA Slow Things Down?

Classic QLoRA saves a lot of memory by storing the frozen base weights ($W_0$) in low precision (INT8 / NF4). However, standard Backpropagation still needs two big matrix multiplications on every single training step:

1. **Forward Pass:** $y = x W_0 + x A B$
2. **Backward Pass:** $\delta_{l-1} = \delta_l W_0^T + \dots$

Even if you can multiply $W_0$ super fast in the forward pass using INT8 VNNI instructions, **accessing $W_0^T$ in the backward pass** forces you into one of two bad choices:

* **Single-copy QLoRA:** Dequantize $W_0$ to FP32 on-the-fly on every step and multiply the transpose. This makes the backward pass heavy and slow on CPU.
* **Double-copy QLoRA:** Keep both $W_0$ and $W_0^T$ separately in memory as INT8. This throws away half of your $W_0$ memory savings, and at best only brings the backward pass down to the forward pass speed.

**The main idea of SAP-QLoRA:** Never read $W_0^T$ in the backward pass at all. Instead, project the output error ($e \in \mathbb{R}^{V}$) directly to each layer using small projection matrices ($P_l \in \mathbb{R}^{V \times D}$) that are aligned with the frozen $W_0$ and current activation masks every $K$ steps.

---

## 2. Architecture and Algorithm

### 2.1. Self-Aligning Projection (SAP)

For each layer $l$, instead of flowing back through the $W_0^T$ chain, the error signal is projected straight from the output error:

$$\delta_l = (e \, P_l) \odot m_l$$

* $e \in \mathbb{R}^{B \times V}$: Output layer error.
* $P_l \in \mathbb{R}^{V \times D}$: Aligned projection matrix updated every $K = 100$ steps (plus an 8-batch masked warm-start at the beginning) using $W_0$, $A$, $B$, and the average ReLU mask $m_l$.
* $W_0^T$ is never read inside the training loop. It is only used during the lightweight sweep every $K$ steps to refresh $P_l$.

### 2.2. AVX-512 VNNI Forward Pass Kernel

* Frozen $W_0$ weights are quantized to INT8 with per-row scaling (`W0_scale`) and row sums (`W0_row_sum`).
* In the forward pass, activations are dynamically converted to INT8 and multiplied directly using **AVX-512 VNNI** (`_mm512_dpbusd_epi32`) for `INT8 × INT8 → INT32`.
* The kernel is verified bit-for-bit against a scalar reference implementation.

---

## 3. Test Setup and Compared Modes

* **Speed Profile Settings:** `warmup = 10`, `measure = 100`, `K = 100`, `rank = 16`, validation (`val`) off.
* **Vocabulary Size:** Character-level language model ($V = 65$).

The 4 modes I compared:

1. **`BP` (`BP-LoRA`):** FP32 $W_0$, standard Backprop (AVX2 FMA for forward and backward).
2. **`BPQ` (`BP-QLoRA`):** Fair single-copy INT8 baseline. Forward pass uses the exact same AVX-512 VNNI as `SAP-QLoRA`. Backward pass dequantizes $W_0$ to FP32 on-the-fly in 128-wide panels and feeds it into the same AVX2 FMA kernel as `BP-LoRA`. *(Self-test confirmed it matches FP32 $W_0^T$ bit-for-bit.)*
3. **`SAP` (`SAP-LoRA`):** FP32 $W_0$, using SAP projection instead of $W_0^T$ in the backward pass.
4. **`QL` (`SAP-QLoRA`):** INT8 $W_0$ (AVX-512 VNNI forward pass) + SAP backward pass.

---

## 4. Speed, Memory, and Breakdown Results ($D = 256 \to 2048$)

### 4.1. Detailed Time (ms/step) and Memory (MB) Table

| $D$ | Mode | Fwd (ms) | Out (ms) | Bwd (ms) | dA/dB (ms) | Dequant (ms) | Update (ms) | **Total (ms)** | **Total (MB)** | **$W_0$ (MB)** |
|---|---|---|---|---|---|---|---|---|---|---|
| **256** | BP | 14.80 | 2.07 | 17.32 | 2.09 | 0.35 | 0.34 | **36.98** | 22.08 | 1.50 |
| **256** | BPQ | 12.03 | 2.03 | 16.99 | 2.00 | 0.34 | 0.33 | **33.73** | 20.97 | 0.39 |
| **256** | SAP | 14.88 | 2.08 | 8.98 | 2.12 | 0.51 | 0.34 | **28.93** | 22.53 | 1.50 |
| **256** | QL | 12.60 | 2.08 | 8.88 | 2.03 | 0.50 | 0.35 | **26.45** | 21.42 | 0.39 |
| **512** | BP | 42.68 | 3.06 | 51.22 | 4.16 | 0.55 | 0.67 | **102.35** | 45.28 | 6.00 |
| **512** | BPQ | 29.10 | 3.06 | 50.31 | 4.05 | 0.54 | 0.67 | **87.75** | 40.80 | 1.52 |
| **512** | SAP | 43.00 | 3.13 | 18.06 | 4.04 | 0.78 | 0.70 | **69.72** | 46.18 | 6.00 |
| **512** | QL | 30.19 | 3.17 | 18.50 | 4.18 | 0.78 | 0.72 | **57.56** | 41.70 | 1.52 |
| **1024** | BP | 167.63 | 6.95 | 187.88 | 13.16 | 1.02 | 1.49 | **378.15** | 100.66 | 24.00 |
| **1024** | BPQ | 79.81 | 6.90 | 163.01 | 10.82 | 0.99 | 1.35 | **262.91** | 82.71 | 6.05 |
| **1024** | SAP | 148.28 | 6.74 | 41.34 | 10.46 | 1.13 | 1.44 | **209.41** | 102.46 | 24.00 |
| **1024** | QL | 77.75 | 6.81 | 42.57 | 10.99 | 1.14 | 1.74 | **141.02** | 84.51 | 6.05 |
| **2048** | BP | 590.93 | 13.01 | 607.81 | 23.74 | 2.25 | 2.72 | **1240.48** | 247.43 | 96.00 |
| **2048** | BPQ | 203.93 | 13.21 | 528.95 | 24.12 | 2.34 | 2.74 | **775.31** | 175.52 | 24.09 |
| **2048** | SAP | 598.66 | 12.80 | 94.07 | 22.80 | 2.18 | 3.05 | **733.57** | 251.03 | 96.00 |
| **2048** | QL | 198.96 | 12.64 | 94.19 | 22.96 | 2.27 | 4.46 | **335.52** | 179.12 | 24.09 |

### 4.2. Speedup Breakdown: VNNI Share vs. Pure SAP Share

Here is the direct answer to *"How much of the 3.70× speedup comes from quantization/VNNI, and how much comes from SAP?"*:

| $D$ | VNNI Share (`BP / BPQ`) | Pure SAP Share (`BPQ / QL`) | **Total Speedup (`BP / QL`)** |
|---|---|---|---|
| **256** | 1.10× | 1.28× | **1.40×** |
| **512** | 1.17× | 1.52× | **1.78×** |
| **1024** | 1.44× | 1.86× | **2.68×** |
| **2048** | 1.60× | **2.31×** | **3.70×** |

### 4.3. System Analysis: L3 Cache Cliff and Memory Trade-off

* **SAP's contribution is bigger than VNNI at every scale:** At $D = 2048$, out of the `905 ms` saved going from `BP-LoRA` to `SAP-QLoRA`, `465 ms` comes from the INT8 forward pass (`BP → BPQ`), and `440 ms` comes from the SAP backward pass (`BPQ → QL`).
* **L3 cache threshold ($D = 2048$):** FP32 $W_0$ alone is `96.00 MB`, which overflows the Ryzen 7 7800X3D's `96 MB` L3 cache and spills into main RAM. Meanwhile, INT8 $W_0$ (`24.09 MB`) stays comfortably inside L3. Because of this, the forward pass gets **~3×** faster (`591 → 199 ms`) and the backward pass gets **~6.4×** faster (`608 → 94 ms`).
* **Memory trade-off:** `SAP-QLoRA` (`179.12 MB`) only uses **`+3.60 MB` (`+2.0%`)** more memory than single-copy `BP-QLoRA` (`175.52 MB`) for the $P_l$ matrices, and gives you a **2.31×** speedup in return.
* **Note on double-copy `BP-QLoRA`:** Even if you kept a transposed `W0_T_int8` copy in memory to speed up the backward pass, the backward pass would at best drop to the forward pass speed (`~199 ms`). SAP's `94.19 ms` backward pass would still be **~2.1×** faster, and `BP-QLoRA`'s $W_0$ memory would double from `24 MB` to `48 MB` (around `~200 MB` total), making it worse than `SAP-QLoRA` in memory too.

---

## 5. Accuracy (BPC) and Ablation Studies

### 5.1. $D = 256$ Accuracy and Mask Ablation (2000 Steps)

* **Frozen base:** `3.016 val BPC`, exact same minibatch order.

| Variant | Step 1000 `val BPC` | Step 2000 `val BPC` | Gap vs Reference |
|---|---|---|---|
| **BP-LoRA** | `2.607` | `2.628` | Reference |
| **SAP-QLoRA (base, unmasked)** | **`2.641`** | **`2.674`** | **`+0.034`** (step 1000) / `+0.046` (step 2000) |
| **+ Masked init (`prime`)** | `2.651` | `2.689` | `+0.015` (worse than base) |
| **+ EMA mask** | `2.652` | `2.682` | `+0.008` (worse than base) |

At $D = 256$, `SAP-QLoRA` tracks `BP-LoRA` with a gap of just **`+0.034 BPC`**.

### 5.2. $D = 1024$ Accuracy, LR Schedule, and Fair Check (1000 Steps)

* **Frozen base:** `build\w0_d1024.bin` (frozen at step 500, `3.216 val BPC`), same minibatch order.

#### Initial Run (Fixed/High LR, Unmasked Init)
* `BP-LoRA` hit **`2.871`** at step 900 (`2.907` at step 1000).
* Unmasked `SAP-QLoRA` jumped backwards to **`3.369`** at step 100 (worse than the `3.216` base), wasted the first 300 steps recovering from that spike, and only reached **`3.094`** at best (step 700, `3.193` at step 1000), leaving a **`+0.223 BPC`** gap.

#### Run 1, Run 2, and Fair `BP-LoRA` Check (Shared Warmup + Cosine LR)
To fix the step-100 spike and oscillation, I added an **8-batch masked init (`prime`)** and a **shared LR schedule** (100-step linear warmup: `3e-5 → 0.0015`, 900-step cosine decay: `0.0015 → 0.00015`). For an apples-to-apples comparison, I ran `BP-LoRA` with the exact same schedule:

| Step | Run 1 (`SAP-QLoRA` + prime) | Run 2 (+ running-mean & `K=20→100`) | **BP-LoRA (Same LR)** | **Same-Schedule Gap** |
|---|---|---|---|---|
| **100** | 3.09909 | 3.10073 | 3.09904 | **`+0.002`** |
| **200** | 3.06484 | 3.06431 | 3.06202 | **`+0.002`** |
| **300** | 3.00820 | 3.00595 | 2.99670 | `+0.009` |
| **400** | 3.00031 | 2.99878 | 2.98217 | `+0.017` |
| **500** | 3.00367 | 3.00261 | 2.98330 | `+0.019` |
| **600** | 2.98676 | 2.98634 | 2.96232 | `+0.024` |
| **700** | 2.98403 | 2.98261 | 2.95431 | `+0.028` |
| **800** | 3.00039 | 3.00064 | 2.96920 | `+0.031` |
| **900** | 2.98030 | 2.98014 | 2.94722 | `+0.033` |
| **1000** | **2.97582** | **2.97539** | **2.93981** | **`+0.036`** |

* **Main takeaway:** Under the same LR schedule, the $D = 1024$ gap is **`+0.036 BPC`**, which is almost identical to the `+0.034 BPC` gap at $D = 256$. Scaling $D$ up by 4× did not widen the accuracy gap.
* **Honest note:** The reason `BP-LoRA` reached `2.871` (`+0.105` gap) in the old fixed/high LR run is that full Backprop can handle higher learning rates without oscillating.

---

## 6. Time-to-Accuracy ($D = 1024$)

Speed per step alone isn't enough. What really matters is the validation BPC reached in the same wall-clock time. Under the shared LR schedule (`SAP-QLoRA = 141.02 ms/step`, `BP-LoRA = 378.15 ms/step`):

| Wall-Clock Time | BP-LoRA (Step / BPC) | SAP-QLoRA (Step / BPC) | Winner on Time Axis |
|---|---|---|---|
| **~38–42 s** | ~step 100 / `3.099` | ~step 300 / **`3.006`** | **SAP-QLoRA (`-0.093 BPC`)** |
| **~70–75 s** | ~step 200 / `3.062` | ~step 500 / **`3.003`** | **SAP-QLoRA (`-0.059 BPC`)** |
| **~100–113 s** | ~step 300 / `2.997` | ~step 700 / **`2.983`** | **SAP-QLoRA (`-0.014 BPC`)** |
| **~141 s** | ~step 375 / `~2.985` | step 1000 / **`2.975`** | **SAP-QLoRA (`-0.010 BPC`)** |

* **Time to reach `3.00 BPC`:** `SAP-QLoRA` takes **~42 seconds**, while `BP-LoRA` takes **~104–113 seconds** (**~2.4× – 2.7×** faster to the same accuracy).

---

## 7. Depth Test ($D = 256, L = 12$)

Most Backpropagation alternatives (DFA, Forward-Forward, PEPITA, etc.) scale fine with width but break down when you add depth ($L$). To test this limit, I doubled the number of blocks ($L = 6 \to 12$).

* **Frozen base:** `build\w0_d256_l12.bin` (frozen at step 1900, `3.061 val BPC`).
* **Settings:** Shared Run 1 warmup + cosine LR schedule; masked init + fixed $K = 100$ for `SAP-QLoRA`.

| Step | BP-LoRA (`val BPC`) | SAP-QLoRA (`val BPC`) | Gap |
|---|---|---|---|
| **100** | 2.94306 | 2.95590 | `+0.013` |
| **200** | 2.88638 | 2.90426 | `+0.018` |
| **300** | 2.84053 | 2.85725 | `+0.017` |
| **400** | 2.82422 | 2.84268 | `+0.018` |
| **500** | 2.83678 | 2.86251 | `+0.026` |
| **600** | 2.81687 | 2.84207 | `+0.025` |
| **700** | **2.80911** | 2.83886 | `+0.030` |
| **800** | 2.83166 | 2.85769 | `+0.026` |
| **900** | 2.81979 | 2.84574 | `+0.026` |
| **1000** | 2.81385 | **2.83875** | **`+0.025`** |

* **Best-point gap:** **`+0.030 BPC`** (`2.809` vs `2.839`). The `+0.034` gap from 6 layers did not grow at all at 12 layers.
* **Gradient health:** At step 1, $\|\delta_0\| = 0.00354$, $\|\delta_5\| = 0.00337$, $\|\delta_{11}\| = 0.00182$ (**ratio `1.95`**). At step 500, the ratio is **`1.89`**. No exploding or vanishing gradients across 12 layers.
* **Init alignment cosine:** `block-11 = 1.000`, `block-0 = 0.335` (the signal travels cleanly back through all 12 layers).

---

## 8. Honest Limits and Scope ($V \ll D$ Regime)

SAP-QLoRA's speed and memory advantage depends on the projection matrices ($P_l \in \mathbb{R}^{V \times D}$) being much smaller than the main weight matrices ($W_0 \in \mathbb{R}^{D \times D}$), which means **$V \ll D$**.

### Why Doesn't It Work Directly on Standard BPE LLMs?
In modern LLMs, the vocabulary size $V$ is 15 to 71 times larger than the hidden dimension $D$:

| Model | $V$ (Vocab) | $D$ (Hidden Dim) | $V / D$ Ratio |
|---|---|---|---|
| **GPT-2 Small** | 50,257 | 768 | 65.4× |
| **Llama 3 8B** | 128,256 | 4,096 | 31.3× |
| **Llama 3 70B** | 128,256 | 8,192 | 15.6× |
| **Qwen 2.5 7B** | 152,064 | 3,584 | 42.4× |
| **Gemma 2 9B** | 256,000 | 3,584 | 71.4× |

Building $P_l$ directly from logit space in these models would make $P_l$ dozens of times bigger than $W_0$.

### Real-World Workloads Where SAP-QLoRA Naturally Shines
1. **Byte-Level Language Models ($V = 256$):** In architectures like ByT5, MegaByte, and BLT, $V = 256$ is fixed. At $D = 1024 \dots 4096$, $V \ll D$ holds naturally.
2. **RLHF Reward Models ($V = 1$) and Classifiers ($V = 2 \dots 100$):** When fine-tuning a big backbone ($D = 2048 \dots 4096$) for a single scalar reward score or a classification head, the output size is tiny, so SAP-QLoRA gives maximum speed and memory benefits right out of the box.

---

## 9. Discarded Ideas (Negative Results Log)

To keep the code clean and avoid unnecessary complexity, I dropped ideas that sounded good in theory but didn't give measurable gains in practice:

* **EMA Mask ($D = 256$):** Ended up `+0.008 BPC` worse (`2.682`) than base `SAP-QLoRA` (`2.674`).
* **Running-Mean Mask + Dynamic Sweep (`K = 20 → 100`) ($D = 1024$):** Only made a `0.0004 BPC` difference (`2.97539`) compared to Run 1 (`2.97582`), so I kept fixed `K = 100`.
* **dA/dB Gradient Clipping (`norm > 1.0`):** Never triggered once in 1000 steps (`ab-clip = 0`).
* **Per-Layer $P_l$ Normalization:** Not needed since gradient norm ratios stayed healthy on their own (`2.2×` at $D = 1024$ and `1.89×` at $L = 12$).

---

## 10. Deep Dive

### 10.1. Where Does ReLU Actually Sit? (Full Step-by-Step Data Flow)

To understand where everything fits, here is the exact path of data through **one layer ($l$)** of the network:

1. **Input ($x$):** Numbers coming from the previous layer (size $B \times D$).
2. **Linear Layer + LoRA:**
   $$y = x W_0 + x A B$$
   We multiply the input by the frozen base weights ($W_0$) and add the small trainable LoRA branch ($x A B$)[cite: 1].
3. **ReLU Activation (Right Here!):**
   $$a_l = \text{ReLU}(y) = \max(0, y)$$
   Every negative number in $y$ is turned into `0`, and every positive number stays untouched.
   * At the exact same time, we record the **Activation Mask ($m_l$)**: `1` if $y > 0$, `0` if $y \le 0$.
   * **Why does ReLU exist?** Without a non-linear function like ReLU between layers, stacking 12 linear matrix multiplications would mathematically collapse into just 1 big matrix multiplication. ReLU is what lets a deep network learn complex patterns.
4. **Next Layer:** The ReLU output $a_l$ becomes the input $x$ for layer $l+1$.
5. **Output Layer (After all $L$ layers):** Projects the final $D$-dimensional vector to size $V$ (**logits** — raw scores for each of the $V = 65$ characters)[cite: 1]. Comparing these logits to the true next character gives the **Output Error ($e$)**[cite: 1].
6. **Backward Pass (Where $m_l$ comes back):** When the error signal comes back to layer $l$ (via $W_0^T$ in Backprop, or via $e P_l$ in SAP)[cite: 1], we multiply it element-by-element by the ReLU mask $m_l$ ($\odot m_l$)[cite: 1]. Why? Because if a neuron was `0` (off) during the forward pass, its derivative is `0` — it didn't contribute to the output, so it shouldn't get a gradient update.

---

### 10.2. Every Symbol & Letter in the Formulas

* **$D$ (Hidden Dimension / Width):** Size of each layer (`256`, `512`, `1024`, `2048`)[cite: 1]. The base weight matrix $W_0$ has size $D \times D$[cite: 1].
* **$L$ (Depth):** Number of stacked layers/blocks (`6` or `12`)[cite: 1].
* **$V$ (Vocabulary Size):** Number of possible output classes. In my character-level model, $V = 65$[cite: 1].
* **$B$ (Batch Size):** How many training samples are processed together in one step[cite: 1].
* **$r$ (`rank = 16`):** The tiny inner dimension of the LoRA matrices $A$ ($D \times r$) and $B$ ($r \times D$)[cite: 1].
* **$W_0$ and $W_0^T$ (Transpose):** $W_0$ is the frozen base weight matrix[cite: 1]. $W_0^T$ is its **transpose** (flipping rows and columns). Backprop needs $W_0^T$ to push gradients backward[cite: 1], which ruins memory access patterns and forces dequantization on CPU[cite: 1].
* **$e$ (Output Error):** A $B \times V$ matrix holding the difference between the model's prediction and the ground truth at the very end of the network[cite: 1].
* **$P_l$ (Projection Matrix):** A small $V \times D$ matrix for layer $l$ used by SAP to project $e$ directly to layer $l$ without touching $W_0^T$[cite: 1].
* **$\delta_l$ (Layer Gradient):** The error signal at layer $l$, computed in SAP as $\delta_l = (e \, P_l) \odot m_l$[cite: 1].
* **$\odot$ (Hadamard Product):** Element-wise multiplication (just `out[i] = a[i] * b[i]`).
* **$\|\delta_l\|$ (Gradient Norm):** The length/magnitude (L2 norm) of the gradient vector at layer $l$, used to check if gradients are exploding or vanishing across layers[cite: 1].
* **$K$ (Sweep Period, `K = 100`):** How often SAP refreshes $P_l$[cite: 1].

---

### 10.3. Every Table Column Explained (`ms/step` & `MB`)

In Section 4.1, each training step is broken down into 6 timing buckets[cite: 1]:

* **`Fwd` (Forward Pass):** Time spent running $x W_0 + x A B$ + ReLU across all $L$ layers[cite: 1].
* **`Out` (Output Head):** Time spent computing the final logits ($V = 65$), loss, and output error $e$[cite: 1].
* **`Bwd` (Backward Pass):** Time spent computing layer gradients $\delta_l$ — multiplying by $W_0^T$ in `BP`/`BPQ`, or multiplying by $P_l$ in `SAP`/`QL`[cite: 1]. This is where SAP cuts `528.95 ms` down to `94.19 ms` at $D = 2048$[cite: 1].
* **`dA/dB`:** Time spent computing the actual weight gradients for the trainable LoRA matrices $A$ and $B$[cite: 1].
* **`Dequant`:** Overhead of dynamic quantization / dequantization conversions[cite: 1].
* **`Update`:** Time spent updating $A$ and $B$ weights (optimizer step), plus the amortized cost of SAP's periodic $P_l$ sweep[cite: 1].
* **`Total (MB)` vs. `$W_0$ (MB)`:** `$W_0$` is just the frozen base weights; `Total` includes everything (activations, gradients, LoRA matrices, optimizer states, and $P_l$ matrices)[cite: 1].

---

### 10.4. Why Intrinsics? (SIMD, AVX2, AVX-512 VNNI)

Normally, when you write a `for` loop in C++, you hope the compiler optimizes it. For quantized matrix math, compilers usually fail to generate optimal assembly. **SIMD Intrinsics** (Single Instruction, Multiple Data) are C++ functions that map **1-to-1 to exact CPU assembly instructions**, letting me control the vector registers directly.

* **Data Types (`FP32`, `INT8`, `INT32`, `NF4`):**
  * **FP32:** Standard 32-bit float (4 bytes per number).
  * **INT8:** 8-bit integer (1 byte per number) — shrinks $W_0$ memory by **4×**[cite: 1].
  * **INT32:** 32-bit integer accumulator so summing hundreds of `INT8 × INT8` products doesn't overflow[cite: 1].
  * **NF4:** 4-bit NormalFloat, commonly used in GPU QLoRA implementations[cite: 1].
* **AVX2 FMA (`BP-LoRA` & `BP-QLoRA` Backward Pass):** Uses 256-bit registers (`__m256`) holding **8 FP32 floats**. FMA (Fused Multiply-Add) computes `(a * b) + c` on all 8 floats in a single instruction[cite: 1].
* **AVX-512 VNNI (`SAP-QLoRA` & `BP-QLoRA` Forward Pass):** Uses 512-bit registers (`__m512i`) holding **64 bytes**. The instruction `_mm512_dpbusd_epi32` takes 64 unsigned `uint8` values, multiplies them by 64 signed `int8` values, sums them in groups of 4, and accumulates into 16 `int32` lanes — **all in one CPU instruction**[cite: 1].
* **Why `W0_scale` and `W0_row_sum` exist:** Because `_mm512_dpbusd_epi32` requires one unsigned input (`0..255`) and one signed input (`-128..127`), we shift activations by `+128`. That shift adds an extra `128 * sum(row_of_W0)` to the dot product. Precomputing `W0_row_sum` once lets us subtract that offset cleanly and multiply by `W0_scale` to get the exact FP32 result[cite: 1].
* **Scalar Reference & Bit-for-Bit Verification:** Every SIMD kernel was checked against a plain, unoptimized scalar C++ loop to guarantee the output matches **bit-for-bit**[cite: 1].

---

### 10.5. CPU Cache & Memory Architecture

* **L1 / L2 / L3 Cache vs. DDR5 RAM:** Caches are tiny, ultra-fast memory inside the CPU; RAM is huge but much slower. My Ryzen 7 7800X3D has **96 MB of L3 cache** (3D V-Cache)[cite: 1].
* **L3 Cache Cliff ($D = 2048$):** At $D = 2048$, FP32 $W_0$ alone is `96.00 MB`, which overflows the 96 MB L3 cache and spills into slow DDR5 RAM[cite: 1]. INT8 $W_0$ is `24.09 MB` and stays inside L3[cite: 1]. That's why speedups jump sharply at $D = 2048$[cite: 1].
* **128-Wide Panels (On-the-Fly Dequantization):** In single-copy `BP-QLoRA`, instead of unpacking all of $W_0$ into RAM, we unpack small 128-wide strips directly into L1 cache, run AVX2 FMA, and discard them[cite: 1].
* **Single-Copy vs. Double-Copy QLoRA:** Keeping a second transposed copy `W0_T_int8` in RAM doubles $W_0$ memory (`24 → 48 MB`) and still can't beat SAP's `94.19 ms` backward pass[cite: 1].

---

### 10.6. Training, Gradients, and Evaluation Concepts

* **Gradient & Backpropagation (BP):** A gradient tells each weight which direction to move to reduce the error. Backprop uses the chain rule to compute gradients by walking backward through $W_0^T$ at every layer[cite: 1].
* **Exploding / Vanishing Gradients:** In deep models ($L = 12$), gradients can shrink to `0` or blow up to infinity. Our bottom-to-top ratio $\|\delta_0\| / \|\delta_{11}\|$ stayed at `1.95` (step 1) and `1.89` (step 500) — completely stable[cite: 1].
* **Masked Init (`prime`) & Alignment Cosine:** Running 8 batches before step 1 to measure real ReLU masks and pre-align $P_l$ fixed the step-100 spike at $D = 1024$[cite: 1]. **Alignment cosine** measures how closely SAP's gradient direction matches true Backprop (`1.0` = identical; at $L = 12$, `block-11 = 1.000` and `block-0 = 0.335`)[cite: 1].
* **BPC (Bits Per Character):** Standard loss metric for character-level language models. Lower is better[cite: 1].
* **Warmup (Two Different Things!):**
  * *Benchmark warmup (`warmup = 10`):* Running 10 dummy steps before timing so CPU caches and boost clocks stabilize[cite: 1].
  * *LR warmup (`100 steps`):* Starting with a tiny learning rate (`3e-5`) and ramping up to `0.0015` so early steps don't overshoot[cite: 1].
* **Cosine Decay:** Smoothly lowering the learning rate (`0.0015 → 0.00015`) along a cosine curve so the model settles into a low BPC[cite: 1].
* **Minibatch & Validation (`val`):** Training on small chunks of data in a locked identical order for fair comparisons, and evaluating on unseen validation data (`val BPC`)[cite: 1].
* **Time-to-Accuracy (Wall-Clock Time):** Measuring real seconds on the clock to reach a target BPC (e.g., `3.00 BPC` in `42 s` vs `104–113 s`)[cite: 1].
* **Ablation Study & Discarded Ideas:** Testing one feature at a time and cutting what doesn't help — **EMA Mask** (exponential moving average of masks), **Running-Mean Mask**, **Gradient Clipping (`ab-clip`, norm > 1.0)**, and **Per-Layer $P_l$ Normalization** were all tested and dropped[cite: 1].

---

### 10.7. Other AI Terms Mentioned in the Scope Section

* **DFA / Forward-Forward / PEPITA:** Famous alternative training algorithms that try to replace Backpropagation, which typically struggle when depth ($L$) increases[cite: 1].
* **BPE (Byte-Pair Encoding):** The tokenizer used by modern LLMs (GPT-2, Llama 3, Qwen, Gemma) that splits text into 50,000–256,000 sub-word tokens ($V \gg D$)[cite: 1].
* **Logit Space:** The raw, unnormalized output vector of size $V$ produced by the final layer before converting to probabilities[cite: 1].
* **Byte-Level LMs ($V = 256$):** Models like ByT5, MegaByte, and BLT that read raw bytes directly, keeping $V = 256$ fixed so $V \ll D$ holds naturally[cite: 1].
* **RLHF Reward Models ($V = 1$) & Classifiers ($V = 2 \dots 100$):** Models that take a huge backbone ($D = 2048 \dots 4096$) and output just 1 scalar score (reward) or a few class labels — the ideal real-world use case for SAP-QLoRA[cite: 1].
