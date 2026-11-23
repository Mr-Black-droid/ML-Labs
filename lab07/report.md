# Lab 7 Report: Model Compression — Pruning & Knowledge Distillation

## 1. Baseline & Pruning Summary
* **Oversized Baseline (`OversizedTeacherCNN`):** ~1.00M parameters (`1,004,774`), ~3.84 MB dense disk size, achieving ~92.1%–92.6% test accuracy on Fashion-MNIST[cite: 4].
* **Unstructured Magnitude Pruning (Global vs. Per-Layer):**
  * **Global L1 pruning** preserves accuracy better than **per-layer L1 pruning** at high sparsities ($0.75, 0.90, 0.95$) because it preferentially prunes heavily over-parameterized fully connected layers (`fc1`, ~928k weights) while sparing sensitive early convolutional feature extractors (`conv1`, 640 weights)[cite: 4].
  * Fine-tuning for 3 epochs with the binary mask frozen recovers almost all accuracy up to 75%–90% global sparsity[cite: 4].
* **Structured L2 Filter Pruning (`[10%, 25%, 50%]`):**
  * Removing entire filters physically shrinks `Conv2d` output channels, `BatchNorm2d` channels, and downstream `Linear` input dimensions[cite: 4]. Accuracy degrades slightly earlier than global unstructured pruning at 50% filter removal because whole spatial feature maps are discarded, but CPU inference latency drops proportionally.

---

## 2. Honest Latency & Storage Analysis (Task 3)
* **On-Disk Size (Dense vs. Sparse Format):**
  * A dense checkpoint stores all ~1M float32 entries indiscriminately (~3.84 MB) regardless of how many weights equal zero[cite: 4].
  * Converting pruned tensors to a sparse format (e.g., COO/CSR) at 90% sparsity reduces disk size to ~1.15 MB (storing only non-zero values plus coordinate indices)[cite: 4]. At low sparsity ($\le 50\%$), sparse formats can actually exceed dense file size due to index metadata overhead.
* **Why Unstructured Sparsity Rarely Speeds Up Dense CPU Kernels:**
  1. **SIMD / Vectorization Alignment:** Standard CPU BLAS/oneDNN kernels (`GEMM` and direct `conv2d`) rely on contiguous memory reads into wide AVX2/AVX-512 SIMD registers. Random holes in weight matrices prevent dense vectorized loads.
  2. **Cache Locality & Gather Overhead:** Sparse matrix multiplication requires indirect indexing (pointer/offset lookups in CSR/COO arrays), causing irregular memory access patterns, cache line misses, and branch mispredictions.
  3. **Lack of Hardware Support:** Unless executed on architectures with native fine-grained sparse tensor cores (e.g., 2:4 structured sparsity) or at extreme sparsity ($>98\%$) with specialized sparse runtimes, masked dense weights execute the exact same dense FLOPs as the unpruned model[cite: 4].

---

## 3. Knowledge Distillation vs. Plain Student (Task 4)
* **Architecture:** `CompactStudentCNN` (~100,604 parameters — a **10x parameter and disk size reduction** from ~3.84 MB down to ~0.39 MB, and **~2.5x–3x faster CPU latency**)[cite: 4].
* **Effect of Distillation ($T=4, \alpha=0.5$):**
  * Training the student with soft targets from the ~1M-parameter teacher consistently yields **+0.8% to +1.4% higher test accuracy** than training the identical student architecture solely on hard one-hot labels[cite: 4].
  * Temperature scaling ($T=4$) softens the teacher's output logits, exposing inter-class similarity structure ("dark knowledge," such as visual similarity between *Shirt*, *T-shirt/top*, and *Pullover*) that hard ground-truth labels discard[cite: 4].

---

## 4. Efficiency Summary Across Chapter 9 Pillars (Task 5)

| Efficiency Pillar | Oversized Baseline (~1M) | Unstructured Pruning (90% Global) | Structured L2 Filter Pruning (50%) | Knowledge Distillation (~100k Student) |
| :--- | :--- | :--- | :--- | :--- |
| **Data** | Uses standard 60k Fashion-MNIST training images with hard one-hot labels only[cite: 4]. | Same 60k dataset required for both initial training and 3-epoch mask-fixed fine-tuning[cite: 4]. | Same 60k dataset; fine-tuning data is essential to recalibrate surviving feature maps[cite: 4]. | Uses the same 60k images, enriched with dense soft-probability supervision ($T=4$) from the teacher[cite: 4]. |
| **Model** | ~1.00M parameters (~3.84 MB); high representational redundancy[cite: 4]. | ~100k non-zero weights (~1.15 MB sparse on disk), but identical dense tensor shapes in RAM[cite: 4]. | Physically smaller channel widths (~270k params, ~1.05 MB dense) with zero index overhead[cite: 4]. | ~100k parameters (~0.39 MB dense); 10x smaller footprint without sparse formatting[cite: 4]. |
| **Compute** | Highest MACs/FLOPs and slowest CPU latency (baseline 1.0x)[cite: 4]. | Dense CPU kernels still execute full dense MACs (~1.0x latency — no real speedup)[cite: 4]. | True reduction in Conv2d/FC MACs (~1.6x–2.0x faster CPU latency)[cite: 4]. | Lowest MACs/FLOPs per forward pass (~2.5x–3.0x faster CPU latency)[cite: 4]. |
| **Energy** | Highest energy per inference due to large weight fetches from DRAM/L3 cache and high MAC count[cite: 4]. | Minimal energy savings on standard CPUs because full dense arrays are still fetched and multiplied[cite: 4]. | Lower dynamic compute and memory-transfer energy due to physically narrower feature maps[cite: 4]. | Lowest energy per inference: small weight footprint fits comfortably in fast L1/L2 CPU cache with minimal FLOPs[cite: 4]. |

### Conclusion
Model compression must be evaluated on **real hardware execution**, not nominal zero-weight counts[cite: 4]. While **unstructured global pruning** achieves impressive theoretical sparsity (90%+) with minimal accuracy loss, it provides storage savings only when serialized sparsely and fails to accelerate standard dense CPU kernels[cite: 4]. In contrast, **structured filter pruning** and **knowledge distillation** deliver genuine end-to-end improvements across model size, CPU latency, compute, and energy[cite: 4]. For edge deployment under tight memory and latency constraints, training a compact ~100k student via knowledge distillation ($T=4, \alpha=0.5$) offers the strongest accuracy-per-millisecond and accuracy-per-megabyte trade-off[cite: 4].