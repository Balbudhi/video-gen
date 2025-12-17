# **Phase 1 Implementation Report: MacBook-Compatible PCA/Semanticist Tokenizer with Head-8 Emphasis**

## **1\. Theoretical Foundations of PCA-Guided Visual Tokenization**

The domain of generative computer vision has historically bifurcated into two distinct methodologies regarding image representation: continuous latent spaces, typified by Variational Autoencoders (VAEs) and later Stable Diffusion, and discrete tokenized spaces, popularized by VQ-VAEs and autoregressive transformers like VQGAN. While discrete tokenization offers the tantalizing promise of unifying vision and language under a single "next-token prediction" paradigm, existing visual tokenizers suffer from a fundamental structural deficiency known as "Semantic-Spectrum Coupling." This phenomenon, where high-level semantic concepts are inextricably entangled with low-level spectral noise across the token sequence, prevents these models from achieving the efficiency and interpretability inherent in classical methods like Principal Component Analysis (PCA).  
The Semanticist project proposes a radical restructuring of visual tokenization.1 Rather than learning an arbitrary codebook mapping that scatters semantic information across a 2D grid of tokens (as seen in VQGAN or TiTok), Semanticist enforces a 1D causal ordering. This ordering is designed to mimic the variance-maximization property of PCA: the earliest tokens in the sequence must capture the "principal components" of the image—its global structure, dominant objects, and semantic class—while subsequent tokens progressively refine high-frequency details. This report details the implementation of "Phase 1" of this architecture, specifically adapted for local development on Apple Silicon hardware.

### **1.1 The Semantic-Spectrum Coupling Problem**

To understand the necessity of the Phase 1 implementation, specifically the "head-8 emphasis," one must first deconstruct the failure modes of standard tokenizers. In models like VQGAN, the tokens correspond to spatial patches. A token at position $(i, j)$ encodes the information for that specific patch. There is no global hierarchy; the first token is no more semantically significant than the last. Consequently, autoregressive models trained on these sequences must expend capacity learning local textures (the "spectrum") alongside global shapes (the "semantics") at every step.  
This coupling leads to inefficient scaling laws. As demonstrated in comparative spectral analyses 1, increasing the token count in standard models like TiTok raises both semantic fidelity and spectral power simultaneously. There is no separation. In contrast, the Semanticist architecture aims for a "coarse-to-fine" hierarchy. Ideally, reconstruction using only the first $k$ tokens should yield a low-pass filtered, semantically accurate approximation of the image. As $k$ increases, the reconstruction should gain high-frequency detail without altering the fundamental semantic content established by the early tokens.

### **1.2 The Role of Prefix Masking in Enforcing Hierarchy**

The mechanism used to enforce this hierarchy is not a new loss function, but a specific conditioning strategy during the training of the diffusion decoder. The architecture utilizes a frozen VAE to encode the image into continuous latents, a ViT encoder to produce slot tokens, and a DiT (Diffusion Transformer) decoder to reconstruct the latents conditioned on the slots.  
The hierarchy is enforced via **Prefix Masking**. During training, the DiT is conditioned not on the full sequence of $N$ slots, but on a truncated prefix of length $k$, where $k \< N$. The remaining $N-k$ tokens are replaced by a learnable null\_cond embedding.1

$$\\text{Cond}(i) \= \\begin{cases} \\text{Slot}\_i & \\text{if } i \\le k \\\\ \\text{Null} & \\text{if } i \> k \\end{cases}$$

By stochastically varying $k$, the model learns an implicit curriculum. When $k$ is small, the model must rely on the first few tokens to predict the entire image structure (minimizing the reconstruction loss of the global VAE latents). When $k$ is large, the model can rely on specific later tokens to resolve fine details. This creates the "PCA-like" property where information is front-loaded.

### **1.3 The Phase 1 Objective: Head-8 Emphasis**

The standard implementation of this masking strategy samples $k$ from a uniform distribution $U\[1, N\]$. While theoretically sound in the infinite-data limit, uniform sampling is inefficient for enforcing a strict "head" hierarchy in limited-compute scenarios (like local MacBook training). If $N=128$, the probability of sampling any specific small $k$ (e.g., $k=1$) is roughly 0.7%. This latent "head" region is under-trained relative to the "tail."  
Phase 1 introduces a **Bimodal Sampling Strategy** to correct this. By explicitly biasing the training distribution toward a set of "head anchors" ($k \\in \\{1, 2, 4, 8\\}$), we artificially force the model to prioritize the compression of semantic data into the first octet of tokens. This "head-8 emphasis" is the defining algorithmic characteristic of this implementation phase, ensuring that even with a smaller model and dataset, the PCA-like behavior emerges robustly.

## ---

**2\. The Hardware Paradigm: Apple Silicon and MPS**

Implementing deep learning research on Apple Silicon (M1/M2/M3 chips) requires a fundamental shift in how one approaches device management and memory optimization. Unlike the discrete memory spaces of the x86 \+ NVIDIA paradigm (where data must move over a PCIe bus between CPU RAM and GPU VRAM), Apple Silicon utilizes a Unified Memory Architecture (UMA). While this offers distinct advantages in data throughput, it introduces unique constraints for PyTorch backends.

### **2.1 The Metal Performance Shaders (MPS) Backend**

The original Semanticist codebase is hardcoded for NVIDIA CUDA environments, utilizing torch.cuda calls that will immediately raise AssertionError or RuntimeError on macOS.1 To enable training on a MacBook, we must leverage the Metal Performance Shaders (MPS) backend. MPS maps PyTorch computational graphs to the Metal API, allowing the execution of kernels optimized for the Apple GPU architecture.2  
However, the MPS backend is not a 1:1 replacement for CUDA. It has a narrower set of supported operations and distinct behaviors regarding floating-point precision. For instance, while CUDA has mature support for TensorFloat32 (TF32) on Ampere cards, MPS relies on different matrix multiplication optimizations that must be explicitly managed via torch.set\_float32\_matmul\_precision in newer PyTorch versions.1

### **2.2 Precision Constraints: The BFloat16 Challenge**

A critical requirement for Phase 1 is the use of bf16 (Brain Floating Point) precision. bf16 offers the dynamic range of float32 with the memory footprint of float16, making it ideal for diffusion training where gradients can explode or vanish.  
However, bf16 support on Apple Silicon is bifurcated:

* **M1 Family:** Does not support native bf16 hardware acceleration. PyTorch may emulate it or fall back to float32, potentially causing performance degradation or crashes depending on the specific operation.5  
* **M2/M3 Family:** Supports native bf16 acceleration, allowing for significant speedups and memory savings.5

The Phase 1 implementation must be robust to these hardware variations. While we aim to configure the training for bf16 to maximize the batch size within the UMA constraints, the codebase must handle potential fallbacks gracefully. The device\_utils.py modification detailed in later sections is designed to handle these backend nuances abstractly, isolating the trainer from the hardware specifics.

### **2.3 Unified Memory and Batch Sizing**

On a MacBook, the GPU shares memory with the OS and CPU. A 16GB MacBook Air does not have 16GB of "VRAM" available for training; it might have effectively 8-10GB available after OS overhead. Training a DiT on $256 \\times 256$ images involves large activation maps.

$$\\text{Activation Size} \\approx B \\times C \\times H \\times W \\times \\text{Bytes}$$  
Standard ImageNet training batch sizes (e.g., 256\) are impossible locally. To replicate the training dynamics of the original paper (which used large batches), we must rely heavily on **Gradient Accumulation**. By setting the physical batch size to a low integer (e.g., 4\) and accumulating gradients over many steps (e.g., 32), we simulate a larger effective batch size of 128 without causing Out-Of-Memory (OOM) errors.1 This strategy is central to the configuration engineering of Phase 1\.

## ---

**3\. Architectural Analysis of the Semanticist Model**

Before detailing the code modifications, it is essential to define the specific architecture being deployed in Phase 1\. The Semanticist model is a composite system relying on three distinct neural networks.

### **3.1 The Frozen VAE Encoder**

The first stage of the pipeline is a pre-trained Variational Autoencoder (VAE). Phase 1 utilizes xwen99/mar-vae-kl16.1 This specific VAE is crucial because it compresses the input image $X \\in \\mathbb{R}^{256 \\times 256 \\times 3}$ into a latent representation $Z$.  
Assuming a downsampling factor of $f=16$ (implied by "kl16"), the latent dimensions are:

$$H\_{latent} \= \\frac{256}{16} \= 16, \\quad W\_{latent} \= \\frac{256}{16} \= 16$$

$$Z \\in \\mathbb{R}^{16 \\times 16 \\times C\_{vae}}$$  
The VAE is frozen during training. The DiT's objective is to reconstruct $Z$, not $X$ directly. This significantly reduces the dimensionality of the optimization problem, making local training feasible.

### **3.2 The ViT Slot Encoder**

The second stage is the ViT Encoder. This model takes the raw image $X$ (or features thereof) and maps it to a sequence of $N=128$ slot tokens $S \\in \\mathbb{R}^{N \\times D}$.  
This encoder utilizes a causal masking mechanism (if enc\_causal=True) to ensure that the prediction of slot $S\_i$ effectively depends only on the relevant features for that hierarchical level. In Phase 1, we use a vit\_base\_patch16 backbone.

### **3.3 The DiT Diffusion Decoder**

The third and primary trainable component is the Diffusion Transformer (DiT). Unlike a standard DiT that is conditioned on a single class label or a text embedding, this DiT is conditioned on the variable-length sequence of slot tokens.  
The conditioning injection happens via cross-attention or adaptive layer normalization (AdaLN), depending on the specific block implementation. The core modification in Phase 1 is strictly controlling how many of these slot tokens the DiT is allowed to "see" via the modified NestedSampler.

## ---

**4\. Implementation Strategy: Backend Adaptation**

The first actionable step in Phase 1 is stabilizing the codebase for the MPS backend. The original file semanticist/utils/device\_utils.py contains rigid CUDA assumptions that must be dismantled.

### **4.1 Modifying semanticist/utils/device\_utils.py**

The primary function of this module is to return the correct torch.device object and configure matrix multiplication precision. The rewritten implementation prioritizes a failsafe approach: it attempts to enable high-precision optimizations (set\_float32\_matmul\_precision) but catches exceptions if the backend (like older MPS versions) does not support them.  
**Code implementation:**

Python

\# semanticist/utils/device\_utils.py

import torch

def configure\_compute\_backend():  
    """  
    Backend settings safe across CUDA/MPS/CPU.  
    This function isolates hardware-specific optimizations to prevent  
    crashes on non-NVIDIA hardware.  
    """  
    try:  
        \# Optimizes matrix multiplication precision for potential speedups.  
        \# On CUDA, this controls TensorFloat32.  
        \# On MPS/CPU, this sets the preferred kernel precision where applicable.  
        \# We wrap this in a try-except block because some PyTorch versions  
        \# on specific OS/hardware combos may not expose this API.  
        torch.set\_float32\_matmul\_precision("high")  
    except Exception:  
        pass

    if torch.cuda.is\_available():  
        \# Retain original CUDA optimizations for backward compatibility  
        \# if the user migrates this code back to a cluster.  
        torch.backends.cuda.matmul.allow\_tf32 \= True  
        torch.backends.cudnn.allow\_tf32 \= True  
        torch.backends.cudnn.benchmark \= True  
        torch.backends.cudnn.deterministic \= False

def get\_device():  
    """  
    Robust device selector implementing the priority queue:  
    CUDA (NVIDIA) \-\> MPS (Apple Silicon) \-\> CPU (Fallback)  
    """  
    if torch.cuda.is\_available():  
        return torch.device("cuda")

    \# Check for Metal Performance Shaders (MPS) availability on macOS.  
    \# We check both the module existence and the is\_available() flag  
    \# to support a wide range of PyTorch versions (1.12+).  
    if hasattr(torch.backends, "mps") and torch.backends.mps.is\_available():  
        return torch.device("mps")

    return torch.device("cpu")

### **4.2 Implications of the Modification**

By introducing this abstraction layer, we ensure that the DiffusionTrainer class (located in semanticist/engine/diffusion\_trainer.py) does not need to be rewritten. The trainer simply calls configure\_compute\_backend() and get\_device(). On a MacBook, get\_device() returns device(type='mps'), and PyTorch automatically dispatches subsequent tensor operations to the Metal backend. This satisfies the requirement to "not rewrite the trainer" 1 while solving the crash.

## ---

**5\. Implementation Strategy: Bimodal Sampling Logic**

The algorithmic core of Phase 1 is the enforcement of the "head-8" hierarchy. This is achieved by manipulating the distribution of the prefix length $k$ during training.

### **5.1 The Bimodal Distribution Logic**

The requirement is to emphasize head anchors $k \\in \\{1, 2, 4, 8\\}$ and the full context $k=128$.  
Let $N=128$. We define a probability mass function $P(k)$ such that:

1. **Full Context Regime:** A significant portion of training steps must see the full sequence to ensure the model can eventually achieve perfect reconstruction. We set $P(k=128) \= 0.50$.  
2. **Head Anchor Regime:** To force the "PCA" structure, we must over-sample the start of the sequence. We set $\\sum\_{k \\in \\{1,2,4,8\\}} P(k) \= 0.40$. Since there are 4 anchors, each has $P(k) \\approx 0.10$.  
3. **Tail Regime:** The remaining probability mass ($0.10$) is distributed among the remaining $N-5$ integers.

Comparing this to a Uniform distribution $U$ where $P(k=1) \\approx 0.0078$, our bimodal sampler increases the frequency of $k=1$ by approximately **12.8 times**. This massive re-weighting is what creates the "Head-8 emphasis".1

### **5.2 Modifying NestedSampler in semanticist/stage1/diffuse\_slot.py**

We replace the existing uniform sampler with a configurable class that supports this bimodal logic.  
**Code Implementation:**

Python

import torch  
import torch.nn as nn

class NestedSampler(nn.Module):  
    """  
    Returns a boolean prefix keep-mask of shape (B, K).

    mode="uniform": original behavior (k \~ Uniform{1..K})  
    mode="bimodal": emphasizes head anchors (e.g., 1/2/4/8) and full K=128,  
                    with occasional intermediate tail lengths.  
    """  
    def \_\_init\_\_(  
        self,  
        num\_slots: int,  
        mode: str \= "uniform",  
        head\_anchors=(1, 2, 4, 8),  
        head\_cutoff: int \= 8,  
        p\_full: float \= 0.50,   \# P(k \= K)  
        p\_head: float \= 0.40,   \# P(k in head\_anchors)  
    ):  
        super().\_\_init\_\_()  
        self.num\_slots \= int(num\_slots)  
        \# Register arange as a buffer so it moves to the correct device (MPS/CUDA) automatically  
        self.register\_buffer("arange", torch.arange(self.num\_slots, dtype=torch.int64))

        self.mode \= mode  
        self.head\_cutoff \= int(head\_cutoff)  
        self.p\_full \= float(p\_full)  
        self.p\_head \= float(p\_head)

        \# Convert tuple to tensor for efficient indexing  
        anchors \= torch.tensor(list(head\_anchors), dtype=torch.int64)  
        self.register\_buffer("head\_anchors", anchors)

        \# Robust input validation  
        if self.mode not in ("uniform", "bimodal"):  
            raise ValueError(f"Unknown mode: {self.mode}")

        if not (0.0 \<= self.p\_full \<= 1.0 and 0.0 \<= self.p\_head \<= 1.0 and (self.p\_full \+ self.p\_head) \<= 1.0):  
            raise ValueError("Need 0\<=p\_full,p\_head and p\_full+p\_head\<=1")

        if self.mode \== "bimodal":  
            if self.head\_cutoff \>= self.num\_slots:  
                raise ValueError("head\_cutoff must be \< num\_slots")  
            if torch.any(self.head\_anchors \< 1\) or torch.any(self.head\_anchors \> self.head\_cutoff):  
                raise ValueError("head\_anchors must be in \[1..head\_cutoff\]")

    def sample\_k(self, batch\_size: int, device: torch.device) \-\> torch.Tensor:  
        """  
        Samples the integer lengths 'k' for the batch.  
        """  
        if self.mode \== "uniform":  
            return torch.randint(1, self.num\_slots \+ 1, (batch\_size,), device=device)

        \# Bimodal Sampling Logic  
        \# 1\. Generate random probabilities for the batch  
        r \= torch.rand(batch\_size, device=device)  
        k \= torch.empty(batch\_size, dtype=torch.int64, device=device)

        \# 2\. Define masks for the three regimes based on cumulative probability  
        full \= r \< self.p\_full  
        head \= (\~full) & (r \< (self.p\_full \+ self.p\_head))  
        tail \= \~(full | head)

        \# 3\. Assign k for "Full" regime  
        k\[full\] \= self.num\_slots

        \# 4\. Assign k for "Head" regime  
        if head.any():  
            \# Randomly select one of the anchors (1, 2, 4, 8\)  
            idx \= torch.randint(0, self.head\_anchors.numel(), (int(head.sum().item()),), device=device)  
            k\[head\] \= self.head\_anchors\[idx\]

        \# 5\. Assign k for "Tail" regime  
        if tail.any():  
            \# Sample uniformly from the remaining range (head\_cutoff \+ 1 to N \- 1\)  
            \# Note: We exclude N because that is covered by 'full'  
            low \= self.head\_cutoff \+ 1  
            high \= self.num\_slots   
            k\[tail\] \= torch.randint(low, high, (int(tail.sum().item()),), device=device)

        return k

    def forward(self, batch\_size, device, inference\_with\_n\_slots=-1):  
        if self.training:  
            k \= self.sample\_k(batch\_size, device)  
        else:  
            \# During validation (without specific override), use full context  
            k \= torch.full((batch\_size,), self.num\_slots, dtype=torch.int64, device=device)

        \# Allow manual override for inference visualization (e.g., forcing k=8 to check quality)  
        if inference\_with\_n\_slots\!= \-1:  
            k \= torch.full(  
                (batch\_size,),  
                min(int(inference\_with\_n\_slots), self.num\_slots),  
                dtype=torch.int64,  
                device=device,  
            )

        \# Create boolean mask: True where index \< k  
        \# Shape:  
        return self.arange\[None, :\] \< k\[:, None\]

This implementation adheres strictly to the requirement of not changing the model interface. The NestedSampler still outputs a boolean mask compatible with DiffuseSlot's forward pass, but the underlying statistics of that mask are now heavily skewed to enforce the Phase 1 objective.

## ---

**6\. Implementation Strategy: Component Integration**

With the sampler defined, we must update the DiffuseSlot class in semanticist/stage1/diffuse\_slot.py to accept the new configuration parameters and instantiate the new sampler. This involves modifying the \_\_init\_\_ method.

### **6.1 Wiring Configuration to Logic**

The original DiffuseSlot initialized NestedSampler with only num\_slots. We must extend the signature to accept nest\_mode, nest\_head\_anchors, etc., and pass them through.  
**Code Implementation:**

Python

class DiffuseSlot(nn.Module):  
    def \_\_init\_\_(  
        self,  
        encoder="vit\_base\_patch16",  
        drop\_path\_rate=0.1,  
        enc\_img\_size=256,  
        enc\_causal=True,  
        num\_slots=16,  
        slot\_dim=256,  
        norm\_slots=False,  
        enable\_nest=False,  
        enable\_nest\_after=-1,  
          
        \# \--- NEW: Phase 1 Sampler Configuration \---  
        nest\_mode="uniform",  
        nest\_head\_anchors=(1, 2, 4, 8),  
        nest\_head\_cutoff=8,  
        nest\_p\_full=0.50,  
        nest\_p\_head=0.40,  
        \# \------------------------------------------  
          
        vae="stabilityai/sd-vae-ft-ema",  
        dit\_model="DiT-B-4",  
        num\_sampling\_steps="ddim25",  
        use\_repa=False,  
        repa\_encoder\_depth=8,  
        repa\_loss\_weight=1.0,  
        \*\*kwargs,  
    ):  
        super().\_\_init\_\_()  
          
        \#  
        \#...  
          
        self.num\_slots \= num\_slots  
        self.norm\_slots \= norm\_slots  
        self.num\_channels \= self.encoder.num\_features  
          
        self.encoder2slot \= nn.Linear(self.num\_channels, slot\_dim)  
          
        \# \--- MODIFIED: Instantiate Bimodal Sampler \---  
        self.nested\_sampler \= NestedSampler(  
            num\_slots,  
            mode=nest\_mode,  
            head\_anchors=nest\_head\_anchors,  
            head\_cutoff=nest\_head\_cutoff,  
            p\_full=nest\_p\_full,  
            p\_head=nest\_p\_head,  
        )  
        \# \---------------------------------------------  
          
        self.enable\_nest \= enable\_nest  
        self.enable\_nest\_after \= enable\_nest\_after

        \#

### **6.2 Logic Flow Analysis**

When train\_net.py instantiates the model via instantiate\_from\_config, it reads the YAML parameters and passes them to this constructor.

1. **Config Parse:** The nest\_mode="bimodal" string is passed.  
2. **Sampler Init:** NestedSampler is created in bimodal mode.  
3. **Forward Pass:** During training, DiffuseSlot.forward\_with\_latents calls self.nested\_sampler(...).  
4. **Sampling:** The sampler uses the bimodal logic to generate a mask.  
5. **Conditioning:** DiT receives the masked tokens. If $k=1$, only the first token is visible; tokens 2-128 are null\_cond.  
6. **Gradient Update:** The DiT learns to reconstruct the image from just Token 1\. This forces Token 1 to become a "Principal Component."

## ---

**7\. Configuration Engineering for Constrained Environments**

The success of Phase 1 depends as much on the configuration as on the code. We must define YAML files that respect the memory constraints of the MacBook (MPS) while ensuring training stability.

### **7.1 Infrastructure Config: configs/mps\_1proc.yaml**

Distributed training frameworks like accelerate usually default to spawning multiple processes. On a local machine with Unified Memory, multiple processes contend for the same RAM, often causing thrashing (excessive paging to disk). Therefore, we enforce a single-process setup.  
**Content:**

YAML

compute\_environment: LOCAL\_MACHINE  
distributed\_type: NO  
num\_processes: 1  
num\_machines: 1  
machine\_rank: 0  
use\_cpu: false  
main\_training\_function: main

This configuration effectively tells accelerate to run the script as a standard Python execution, bypassing the overhead of torch.distributed while still setting up the device (MPS) correctly via our patched device\_utils.py.

### **7.2 Training Config: configs/tokenizer\_mps\_256\_128\_head8.yaml**

This file orchestrates the hyperparameters. We must balance batch size against gradient accumulation to achieve stable convergence.  
**Rationale for Key Parameters:**

* **batch\_size: 4**: A DiT-Small at 256px with a ViT encoder and VAE decoder consumes significant activation memory. 4 is a safe upper bound for 16GB-24GB MacBooks to avoid OOM.  
* grad\_accum\_steps: 32: Since the batch size is small, the gradient estimate is noisy. We accumulate gradients over 32 steps before an optimizer update.

  $$\\text{Effective Batch} \= 4 \\times 32 \= 128$$

  This matches the effective batch size of standard research baselines, ensuring valid loss convergence dynamics.  
* **num\_workers: 0**: On macOS, Python multiprocessing can be unstable or slow due to fork vs spawn contexts and memory copying. Setting workers to 0 runs data loading in the main process, which is often faster for local training of this scale.  
* **precision: "bf16"**: Enables BFloat16. This is the "Phase 1" target. If the user's Mac is M1 (non-Pro/Max/Ultra) or on an old OS, they may need to change this to "no", but the default configuration targets modern M2/M3 behavior.  
* **dit\_model: "DiT-S-4"**: We use the "Small" variant of the Diffusion Transformer. The "Base" (B) or "Large" (L) models would drastically slow down training on MPS. "Small" is sufficient to prove the "Head-8" hypothesis.

**Full Config Content:**

YAML

trainer:  
  target: semanticist.engine.diffusion\_trainer.DiffusionTrainer  
  params:  
    num\_epoch: 50  
    valid\_size: 64

    \# Learning Rate and Optimizer  
    blr: 2.5e-5  
    cosine\_lr: True  
    warmup\_epochs: 5  
    max\_grad\_norm: 1.0

    \# Mac-Optimized Data Loading  
    batch\_size: 4  
    grad\_accum\_steps: 32  
    num\_workers: 0          \# Critical for stability on macOS  
    pin\_memory: False       \# False because MPS uses Unified Memory

    \# Precision Strategy  
    precision: "bf16"       \# Fallback to "no" if M1/OS incompatibility arises

    enable\_ema: True  
    save\_every: 1000  
    sample\_every: 200       \# Frequent sampling for qualitative validation

    eval\_fid: False  
    fid\_every: 999999999    \# Disable FID calc (too slow locally)

    test\_num\_slots: 8  
    cfg: 3.0  
    compile: False          \# torch.compile is currently unstable on MPS

    result\_folder: "./output/tokenizer\_mps256\_128\_head8"  
    log\_dir: "./output/tokenizer\_mps256\_128\_head8/logs"

    model:  
      target: semanticist.stage1.diffuse\_slot.DiffuseSlot  
      params:  
        encoder: "vit\_base\_patch16"  
        enc\_img\_size: 256  
        enc\_causal: True

        num\_slots: 128  
        slot\_dim: 32  
        norm\_slots: True

        \# DiT Architecture: Small variant for MPS throughput  
        dit\_model: "DiT-S-4"  
        num\_sampling\_steps: "ddim25"

        vae: "xwen99/mar-vae-kl16"

        enable\_nest: True  
        enable\_nest\_after: \-1

        \# \--- Phase 1: Bimodal Sampler Config \---  
        nest\_mode: "bimodal"  
        nest\_head\_cutoff: 8  
        nest\_head\_anchors:   
        nest\_p\_full: 0.50  
        nest\_p\_head: 0.40  
        \# \---------------------------------------

        use\_repa: False     \# Disable REPA to save VRAM/Compute  
        ckpt\_path: null

    dataset:  
      target: semanticist.utils.datasets.ImageNet  
      params:  
        root:./dataset/celebs\_small  
        split: train  
        aug: randcrop  
        img\_size: 256

    test\_dataset:  
      target: semanticist.utils.datasets.ImageNet  
      params:  
        root:./dataset/celebs\_small  
        split: val  
        aug: centercrop  
        img\_size: 256

## ---

**8\. Operational Workflow and Validation**

With the code patches applied (device\_utils.py, diffuse\_slot.py) and configurations created (mps\_1proc.yaml, tokenizer\_mps\_256\_128\_head8.yaml), the operational workflow is straightforward but requires strict adherence to dataset structuring.

### **8.1 Dataset Preparation**

The training loader expects a standard ImageFolder structure. For Phase 1 validation, a small dataset (e.g., 10k-20k images) is sufficient to observe the emergence of PCA-like tokens.  
**Directory Structure:**

dataset/celebs\_small/  
├── train/  
│   └── 0/  
│       ├── image\_001.jpg  
│       └──...  
└── val/  
    └── 0/  
        ├── image\_001.jpg  
        └──...

A "dummy" class folder 0/ is required because torchvision.datasets.ImageFolder expects class subdirectories.

### **8.2 Execution Command**

The training is launched via accelerate. This command unifies the infrastructure config (single process) and the model config (Phase 1 settings).

Bash

accelerate launch \--config\_file configs/mps\_1proc.yaml train\_net.py \--cfg configs/tokenizer\_mps\_256\_128\_head8.yaml

### **8.3 Validation: The Qualitative Prefix Sweep**

The success of Phase 1 is measured by the distinct behaviors of the model when reconstructing images from different token counts ($k$). The DiffusionTrainer is configured to output samples every 200 steps.  
**Table 1: Expected Reconstruction Characteristics by Token Count ($k$)**

| Token Count (k) | Expected Visual Output | Theoretical Justification |
| :---- | :---- | :---- |
| **1** | **Global Blur:** A vague, blurry blob representing the dominant object's color and rough position. No sharp edges. | Token 1 captures the "First Principal Component"—the direction of maximum semantic variance (global gist). |
| **2** | **Shape Emergence:** Basic separation of foreground vs. background. Shapes (e.g., "round face") become discernible. | Token 2 adds the next orthogonal component of variance, likely spatial layout. |
| **4** | **Class Identity:** The object is recognizable (e.g., "it's a woman," "it's a cat"). Key landmarks (eyes, wheels) appear. | The first 4 tokens encode the core semantic class features. |
| **8 (Head)** | **Structural Completeness:** The image is structurally correct. Pose, identity, and major objects are fixed. Fine textures (hair strands) are missing or smoothed. | This is the "Head Cutoff." The bimodal sampler heavily trains this state, ensuring 8 tokens are sufficient for a "good enough" proxy. |
| **128 (Full)** | **High Fidelity:** Fine details, textures, and background noise are reconstructed accurately. | The "tail" tokens (9-128) capture the high-frequency spectral residuals. |

Failure Mode Indicator:  
If the model produces garbage or identical quality at $k=1$ and $k=8$, the bimodal sampler is likely failing (or enable\_nest is False), causing the model to ignore the token ordering.

## ---

**9\. Conclusion and Future Trajectories**

This implementation report has detailed a comprehensive strategy for deploying the Phase 1 Semanticist tokenizer on MacBook hardware. By refactoring the device management logic to support MPS, utilizing bf16 precision for memory efficiency, and implementing a novel "Bimodal Nested Sampler," we have established a training pipeline that is both hardware-compatible and theoretically robust.  
The "Head-8 Emphasis" strategy is not merely a constraint for local training; it is a validation of the core Semanticist hypothesis. By successfully forcing the model to compress major semantic variance into the first octet of tokens, we prepare the ground for **Phase 2**, where an autoregressive model will learn to predict these highly structured tokens. The result will be a generative system that naturally generates images in a coarse-to-fine manner, mirroring the global precedence of human vision.  
This Phase 1 implementation serves as the proof-of-concept. Once validated on local hardware, the same code (thanks to the robust device\_utils.py and configurable sampler) can be scaled to NVIDIA clusters for ImageNet-scale training without further modification.

#### **Works cited**

1. PCA-code.txt  
2. Accelerated PyTorch training on Mac \- Metal \- Apple Developer, accessed December 16, 2025, [https://developer.apple.com/metal/pytorch/](https://developer.apple.com/metal/pytorch/)  
3. Introducing Accelerated PyTorch Training on Mac, accessed December 16, 2025, [https://pytorch.org/blog/introducing-accelerated-pytorch-training-on-mac/](https://pytorch.org/blog/introducing-accelerated-pytorch-training-on-mac/)  
4. torch.set\_float32\_matmul\_precision — PyTorch 2.9 documentation, accessed December 16, 2025, [https://docs.pytorch.org/docs/stable/generated/torch.set\_float32\_matmul\_precision.html](https://docs.pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html)  
5. Bfloat16 support coming to Apple's Metal and PyTorch \[video\] | Hacker News, accessed December 16, 2025, [https://news.ycombinator.com/item?id=36575443](https://news.ycombinator.com/item?id=36575443)  
6. VAEDecode Mac BFloat16 is not supported on MPS · Issue \#6254 · comfyanonymous/ComfyUI \- GitHub, accessed December 16, 2025, [https://github.com/comfyanonymous/ComfyUI/issues/6254](https://github.com/comfyanonymous/ComfyUI/issues/6254)  
7. PSA: Recent PyTorch nightlies support enough BFloat16 on MPS to run Cascade. \- Reddit, accessed December 16, 2025, [https://www.reddit.com/r/StableDiffusion/comments/1axbjrp/psa\_recent\_pytorch\_nightlies\_support\_enough/](https://www.reddit.com/r/StableDiffusion/comments/1axbjrp/psa_recent_pytorch_nightlies_support_enough/)